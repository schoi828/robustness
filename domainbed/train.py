# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import shutil
import copy

from torchmetrics.classification import MulticlassConfusionMatrix
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import iwildcam
import torch.optim as optim
from domainbed.fmow import FMoW
from domainbed.camelyon17 import Camelyon17

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        values = values.split('|')
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = float(value)

if __name__ == "__main__":
    start_train = time.time()
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="DSPRITES")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--arch', type=str, default="resnet18",
        choices=['resnet18','resnet50','resnet101','vit','mlp'])
    parser.add_argument('--aug', type=str, default="no_aug", choices=['no_aug','imgnet','augmix','randaug','autoaug'])
    parser.add_argument('--dist_type', type=str,
        choices=['CLUSTER', 'SC', 'LDD', 'UDS', 'SC_LDD', 'SC_UDS', 'LDD_UDS', 'SC_LDD_UDS', 'UNIFORM','NONE', 'CORRUPT'])
    parser.add_argument('--pretrain', action='store_true',
        help='pretrain or not (default: False)')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=1,
        help='Seed for everything else')
    parser.add_argument('--data_size', type=int, default=1,
        help='dataset size')
    parser.add_argument('--patience', type=int, default=20,
        help='patience for early stopping')
    parser.add_argument('--steps', type=int, default=10000,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0, help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_cf', action='store_true')
    parser.add_argument('--cos_anneal', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sc_ratio', type=float, default=0.01)
    parser.add_argument('--attr', type=int, default=0, help='attributes 0 for the first attribute, 1 for the second attribute, etc.')
    parser.add_argument('-k', '--kwargs', type=str, action=ParseKwargs)

    args = parser.parse_args()

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None
    if args.seed == 0:
        args.seed = random.randint(0, 1000)
    # set output dir
    args.output_dir = f"{args.output_dir}/{args.dataset}/pre:{args.pretrain}-{args.dist_type}-{args.algorithm}-{args.arch}-{args.aug}-{args.lr}-attr:{args.attr}-size:{args.data_size}_seed{args.seed}"

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.algorithm == 'MLP':
        args.arch = 'mlp'
    elif args.algorithm == 'pretrained' and args.algorithm != 'VIT':
        args.algorithm = 'ERM'

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
        hparams['arch'] = args.arch
        hparams['pretrain'] = args.pretrain

        #if args.arch == 'vit':
        #    hparams['batch_size'] = 1024
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    
    if args.kwargs:
        print('kwargs:', args.kwargs)
        if args.algorithm == 'OccamNets':
            for hp_key in args.kwargs:
                hparams[hp_key]['loss_wt'] = args.kwargs[hp_key]
        else:
            hparams.update(args.kwargs)

        for key in args.kwargs:
            args.output_dir += f"_{key}_{args.kwargs[key]}"

    hparams['lr'] = args.lr

    if os.path.exists(args.output_dir):
        if len(os.listdir(args.output_dir)) >= 6:
            print(f"Skip results : {args.output_dir}")    
            sys.exit()

    print(f"Save results : {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        #dataset = vars(datasets)[args.dataset](args.data_dir,args.test_envs, hparams)
        if hparams['attr'] is not None and args.dist_type != 'UNIFORM':
            attr_dict = hparams['attr'][args.dist_type]
            attributes = attr_dict[args.attr]
        else:
            attributes = None
        train_dataset = vars(datasets)[args.dataset](args.data_dir,dist_type=args.dist_type,dataset_size=args.data_size,aug=args.aug, resize=(args.algorithm == 'VIT'),algo=args.algorithm,ratio=args.sc_ratio, attributes=attributes,hparams=hparams if args.kwargs else None)
        if args.dataset == 'DSPRITES':
            val_dataset = copy.deepcopy(train_dataset)
            val_dataset._imgs = val_dataset.val_imgs
            val_dataset._labels = val_dataset.val_labels
            val_dataset.postprocess_labels()
            val_dataset.split = 'val'
        else:
            val_dataset = vars(datasets)[args.dataset](args.data_dir,dist_type=args.dist_type,dataset_size=args.data_size,aug=args.aug, resize=(args.algorithm == 'VIT'), algo=args.algorithm, split='val',ratio=args.sc_ratio, attributes=attributes,hparams=hparams if args.kwargs else None)
        test_dataset = vars(datasets)[args.dataset](args.data_dir,dist_type=args.dist_type,dataset_size=args.data_size,aug=args.aug, resize=(args.algorithm == 'VIT'), algo=args.algorithm, split='test',ratio=args.sc_ratio, attributes=attributes,hparams=hparams if args.kwargs else None)

    elif 'iwildcam' in args.dataset:

        train_dataset = vars(iwildcam)[args.dataset](root=args.data_dir, split='train',aug=args.aug, algo=args.algorithm,hparams=hparams if args.kwargs else None)
        test_dataset = vars(iwildcam)[args.dataset](root=args.data_dir, split='test',aug=args.aug, algo=args.algorithm,hparams=hparams if args.kwargs else None)
        val_dataset = vars(iwildcam)[args.dataset](root=args.data_dir, split='val', aug=args.aug, algo=args.algorithm,hparams=hparams if args.kwargs else None)

    elif args.dataset == 'FMOW':
        train_dataset = FMoW(root=args.data_dir, split='train',aug=args.aug, algo=args.algorithm,hparams=hparams if args.kwargs else None)
        test_dataset = FMoW(root=args.data_dir, split='test',aug=args.aug, algo=args.algorithm,hparams=hparams if args.kwargs else None)
        val_dataset = FMoW(root=args.data_dir, split='val', aug=args.aug, algo=args.algorithm,hparams=hparams if args.kwargs else None) 

    elif args.dataset == 'CAMELYON17':
        train_dataset = Camelyon17(root=args.data_dir, split='train',aug=args.aug, algo=args.algorithm,hparams=hparams if args.kwargs else None)
        test_dataset = Camelyon17(root=args.data_dir, split='test',aug=args.aug, algo=args.algorithm,hparams=hparams if args.kwargs else None)
        val_dataset = Camelyon17(root=args.data_dir, split='val', aug=args.aug, algo=args.algorithm,hparams=hparams if args.kwargs else None) 
    else:
        
        raise NotImplementedError
    
    if args.algorithm == 'PnD':
        hparams['data_size'] = len(train_dataset)

    in_splits = []
    out_splits = []
    uda_splits = []
    
    env_weights=None
    train_loaders = [InfiniteDataLoader(
        dataset=train_dataset,
        weights=env_weights,
         batch_size=hparams['batch_size'],
        num_workers=train_dataset.N_WORKERS)]

    print('val_dataset' , len(val_dataset), val_dataset)
    val_loaders = [FastDataLoader(
        dataset=val_dataset,
        batch_size=hparams['batch_size'],
        num_workers=test_dataset.N_WORKERS)]

    val_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    val_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    val_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    val_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    test_loaders = [FastDataLoader(
        dataset=test_dataset,
        batch_size=hparams['batch_size'],
        num_workers=test_dataset.N_WORKERS)]

    test_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    test_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    test_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    test_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(train_dataset.input_shape, train_dataset.num_classes,
        len(train_dataset) - len(args.test_envs), hparams)

    cf_matrix = MulticlassConfusionMatrix(num_classes=train_dataset.num_classes)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = None 
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = len(train_dataset)//hparams['batch_size']

    n_steps = args.steps or train_dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or train_dataset.CHECKPOINT_FREQ
    if args.algorithm in ['ADA','ME_ADA']:
        algorithm.train_data = train_dataset
        algorithm.output_dir = args.output_dir
        algorithm.init_dataloader()
        algorithm.init_iter_loader()

    elif args.algorithm in ['BPA','PnD','UBNet']:
        algorithm.output_dir = args.output_dir
        algorithm.init_dataloader(train_dataset,val_dataset)
        algorithm.train_base_model()
        algorithm.init_centroids()
        
    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": train_dataset.input_shape,
            "model_num_classes": train_dataset.num_classes,
            "model_num_domains": len(train_dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))
    
    ###########
    
    earlystopper = misc.EarlyStopper(patience=args.patience)
    early_stop = False
    last_results_keys = None
    cf_train = torch.zeros(train_dataset.num_classes,train_dataset.num_classes)
    acc = 0
    if args.cos_anneal:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(algorithm.optimizer, T_max=100)
    current_epoch = 0
    for step in range(start_step, n_steps):
        if early_stop or acc == 1.0:
            break
        step_start_time = time.time()
        if args.algorithm in ['ADA','ME_ADA']:
            minibatches_device = []  
        elif args.algorithm in ['BPA', 'PnD', 'OccamNets']:
            minibatches_device = [(x.to(device), y.to(device), idx) for x,y,idx in next(train_minibatches_iterator)]
            algorithm.current_epoch = current_epoch
        else:
            minibatches_device = [(x.to(device), y.to(device)) for x,y in next(train_minibatches_iterator)]
        
        uda_device = None
        
        step_vals, cf_train_step = algorithm.update_cf(minibatches_device, uda_device, cf_matrix)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        cf_train+=cf_train_step
        
        if args.cos_anneal and current_epoch < int(step / max(1,steps_per_epoch)):
            current_epoch+=1
            scheduler.step()

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if ((step % checkpoint_freq == 0) or (step == n_steps - 1)) and step > 0:
            train_acc = torch.trace(cf_train)/torch.sum(cf_train)
            results = {
                'step': step,
                'epoch': step / max(1,steps_per_epoch),
                'train_acc': train_acc.item()
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            vals = zip(val_loader_names, val_loaders, val_weights)
            acc, cf_val = misc.accuracy(algorithm, val_loaders[0], weights=None, device=device,cf_matrix=cf_matrix)
            results['val_acc'] = acc
            save_metric = {'cf_train': cf_train, 'cf_val': cf_val}
            early_stop = earlystopper.early_stop(acc, save=save_metric)
            cf_train = torch.zeros(train_dataset.num_classes,train_dataset.num_classes)

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
            if earlystopper.counter == 0 and not early_stop:
                save_checkpoint('model_best_val.pkl')
        if args.cos_anneal and current_epoch == 100:
            break
    val_checkpoint = torch.load(os.path.join(args.output_dir, 'model_best_val.pkl'))
    algorithm.load_state_dict(val_checkpoint["model_dict"])
    test = zip(test_loader_names, test_loaders, test_weights)
    test_acc, cf_test = misc.accuracy(algorithm, test_loaders[0], weights=None, device=device,cf_matrix=cf_matrix)
    misc.save_cf_matrix(cf_test,test_dataset.label_names, args.output_dir, split='test', mode='acc')
    misc.save_cf_matrix(cf_test,test_dataset.label_names, args.output_dir, split='test', mode='count')    
    
    with open(os.path.join(args.output_dir, 'done_testacc_{:.4f}'.format(test_acc)), 'w') as f:
        f.write('done\n')
        f.write("val accuracy: {:.5f}, test accuracy: {:.5f}\n".format(earlystopper.val_metric,test_acc))
        f.write(f'Elapsed Time: {(time.time() - start_train)/3600: .1f} hour\n')
        
    if hasattr(test_dataset,'change_mode'):
        for dist in ['IID', 'OOD']:
            test_dataset.change_mode(dist)
            test_loaders = [FastDataLoader(
                dataset=test_dataset,
                batch_size=hparams['batch_size'],
                num_workers=test_dataset.N_WORKERS)]
            
            test_acc, cf_test = misc.accuracy(algorithm, test_loaders[0], weights=None, device=device,cf_matrix=cf_matrix)
            misc.save_cf_matrix(cf_test,test_dataset.label_names, args.output_dir, split='test', mode='acc', save_name=f'cf_matrix_{dist}_acc.png')
            misc.save_cf_matrix(cf_test,test_dataset.label_names, args.output_dir, split='test', mode='count',save_name=f'cf_matrix_{dist}_count.png')    
            
            with open(os.path.join(args.output_dir, f'done_{dist}_testacc_{test_acc:.4f}'), 'w') as f:
                f.write('done\n')
                f.write("val accuracy: {:.5f}, test accuracy: {:.5f}\n".format(earlystopper.val_metric,test_acc))
                f.write(f'Elapsed Time: {(time.time() - start_train)/3600: .1f} hour\n')

    if earlystopper.save is not None and args.save_cf:
        cf_train = earlystopper.save['cf_train']
        cf_val = earlystopper.save['cf_val']

        misc.save_cf_matrix(cf_train,train_dataset.label_names, args.output_dir, split='train', mode='acc')
        misc.save_cf_matrix(cf_train,train_dataset.label_names, args.output_dir, split='train', mode='count')

        misc.save_cf_matrix(cf_val,val_dataset.label_names, args.output_dir, split='val', mode='acc')
        misc.save_cf_matrix(cf_val,val_dataset.label_names, args.output_dir, split='val', mode='count')

    if args.algorithm in ['ADA', 'ME_ADA'] and args.dataset == 'iwildcam':
        shutil.rmtree(os.path.join(algorithm.train_data.ada_root))
    if not args.save_checkpoint or args.dataset == 'iwildcam':
        os.remove(os.path.join(args.output_dir, 'model_best_val.pkl'))