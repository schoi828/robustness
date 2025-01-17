# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from domainbed.lib import misc
import random
def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)

def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams), f'hparam duplicate exists: {name}'
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
        
    _hparam('arch', 'resnet18', lambda r: r.choice(['resnet18','resnet50','vit','mlp','resnet101']))
    #_hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))
    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if algorithm in ['DANN', 'CDANN']:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    elif algorithm == 'Fish':
        _hparam('meta_lr', 0.5, lambda r:r.choice([0.05, 0.1, 0.5]))

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "Mixup":
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm == "MMD" or algorithm == "CORAL" or algorithm == "CausIRL_CORAL" or algorithm == "CausIRL_MMD":
        #gamma_dict = {'SMALLNORB':0.5, 'DSPRITES':0.3, 'DEEPFASHION':0.1, 'CELEBA':0.1, 'SHAPES3D':1.0,'iwildcam':1.0}
        gamma_dict = {'SMALLNORB':0.5, 'DSPRITES':0.1, 'DEEPFASHION':0.1, 'CELEBA_C':0.1, 'CELEBA_CLUSTER':0.1, 'CELEBA':0.1, 'SHAPES3D':1.0,'iwildcam':1.0, 'FMOW':1.0, 'POVERTY':1.0, 'CAMELYON17':1.0}

        gamma = gamma_dict[dataset]
        _hparam('mmd_gamma', gamma, lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))
        _hparam('n_meta_test', 2, lambda r:  r.choice([1, 2]))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1))

    elif algorithm == "ANDMask":
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm == "IGA":
        _hparam('penalty', 1000, lambda r: 10**r.uniform(1, 5))

    elif algorithm == "SANDMask":
        _hparam('tau', 1.0, lambda r: r.uniform(0.0, 1.))
        _hparam('k', 1e+1, lambda r: 10**r.uniform(-3, 5))

    elif algorithm == "Fishr":
        _hparam('lambda', 1000., lambda r: 10**r.uniform(1., 4.))
        _hparam('penalty_anneal_iters', 1500, lambda r: int(r.uniform(0., 5000.)))
        _hparam('ema', 0.95, lambda r: r.uniform(0.90, 0.99))

    elif algorithm == "TRM":
        _hparam('cos_lambda', 1e-4, lambda r: 10 ** r.uniform(-5, 0))
        _hparam('iters', 200, lambda r: int(10 ** r.uniform(0, 4)))
        _hparam('groupdro_eta', 1e-2, lambda r: 10 ** r.uniform(-3, -1))

    elif algorithm == "IB_ERM":
        _hparam('ib_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "IB_IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))
        _hparam('ib_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "CAD" or algorithm == "CondCAD":
        _hparam('lmbda', 1e-1, lambda r: r.choice([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
        _hparam('temperature', 0.1, lambda r: r.choice([0.05, 0.1]))
        _hparam('is_normalized', False, lambda r: False)
        _hparam('is_project', False, lambda r: False)
        _hparam('is_flipped', True, lambda r: True)
        
    elif algorithm == "Transfer":
        _hparam('t_lambda', 1.0, lambda r: 10**r.uniform(-2, 1))
        _hparam('delta', 2.0, lambda r: r.uniform(0.1, 3.0))
        _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('gda', False, lambda r: True)
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))

    elif algorithm == 'EQRM':
        _hparam('eqrm_quantile', 0.75, lambda r: r.uniform(0.5, 0.99))
        _hparam('eqrm_burnin_iters', 2500, lambda r: 10 ** r.uniform(2.5, 3.5))
        _hparam('eqrm_lr', 1e-6, lambda r: 10 ** r.uniform(-7, -5))

    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.

    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    elif dataset == 'iwildcam' or dataset=='FMOW' or dataset == 'POVERTY':
        _hparam('weight_decay', 0.0001, lambda r: 10**r.uniform(-6, -2))
    else:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))

    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 9)))
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)))
        
    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))

    if algorithm in ['MLP', 'BetaVAE']:
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 4, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    if algorithm == 'BetaVAE':
        _hparam('base_epochs', 50, lambda r: 100)
        _hparam('base_patience', 10, lambda r: 10)

    if algorithm == 'VIT':
        patch_size_dict = {'DSPRITES':4, 'SHAPES3D':4, 'SMALLNORB':12}
        try:
            patch_size = patch_size_dict[dataset]
        except:
            patch_size = 16
        _hparam('patch_size', patch_size, lambda r: patch_size)

    if algorithm == 'L2D':

        _hparam('lr_sc', 10, lambda r: 10)
        _hparam('beta', 0.1, lambda r: 0.1)
        _hparam('alpha1', 1.0, lambda r: 1.0)
        _hparam('alpha2', 1.0, lambda r: 1.0)

    if algorithm == 'BPA':
        _hparam('k', 8, lambda r: 8)
        _hparam('momentum', 0.3, lambda r: 0.3)
        _hparam('feature_dim', 512, lambda r: 512)
        _hparam('exp_step', 0.05, lambda r: 0.05)
        _hparam('update_cluster_iter', 10, lambda r: 10)
        _hparam('base_epochs', 50, lambda r: 50) #50
        _hparam('base_patience', 10, lambda r: 10) 
        _hparam('max_weight', 10, lambda r: 10.0)#100
        
    if algorithm == 'OccamNets':
        _hparam('cam_suppression', {'loss_wt': 0.1}, lambda r: {'loss_wt': 0.1})
        _hparam('exit_gating', {'loss_wt': 1, 'gamma0':3, 'gamma':1, 'train_acc_thresholds':[ 0.5, 0.6, 0.7, 0.8 ], 'min_epochs':1, 'weight_offset':0.1,
                                'balance_factor':0.5}, lambda r: {'loss_wt': 1, 'gamma0':3, 'gamma':1, 'train_acc_thresholds':[ 0.5, 0.6, 0.7, 0.8 ], 'min_epochs':1, 'weight_offset':0.1,
                                'balance_factor':0.5})

    if dataset in ['DSPRITES', 'SHAPES3D', 'SMALLNORB', 'CELEBA', 'DEEPFASHION','iwildcam','FMOW' , 'CAMELYON17', 'POVERTY', 'CELEBA_C', 'CELEBA_CLUSTER']:
        bs = 128
        _hparam('batch_size', bs, lambda r: bs)
        _hparam('lr', 1e-4, lambda r: r.choice([1e-2, 1e-3,1e-4]))
        
        dsprites_attr = {'SC':['obj_color', 'bg_color', 'scale'], 'LDD':['obj_color', 'bg_color', 'scale'],'UDS':['obj_color', 'bg_color', 'scale'],
            'SC_LDD':[['obj_color', 'bg_color'],['obj_color', 'scale'],['bg_color', 'obj_color'],['bg_color', 'scale'],['scale', 'obj_color'],['scale', 'bg_color']],
            'SC_UDS':[['obj_color', 'bg_color'],['obj_color', 'scale'],['bg_color', 'obj_color'],['bg_color', 'scale'],['scale', 'obj_color'],['scale', 'bg_color']],
            'LDD_UDS':[['obj_color', 'bg_color'],['obj_color', 'scale'],['bg_color', 'obj_color'],['bg_color', 'scale'],['scale', 'obj_color'],['scale', 'bg_color']],
            'SC_LDD_UDS':[['obj_color', 'bg_color', 'scale'],['obj_color', 'scale', 'bg_color'],['bg_color', 'obj_color', 'scale'],['bg_color', 'scale', 'obj_color'],['scale', 'obj_color', 'bg_color'],['scale', 'bg_color', 'obj_color']],
            }
        
        deep_attr = {'SC':['texture', 'sleeve', 'fabric'], 'LDD':['texture', 'sleeve', 'fabric'],'UDS':['texture', 'sleeve', 'fabric'],
            'SC_LDD':[['texture', 'sleeve'],['texture', 'fabric'],['sleeve', 'texture'],['sleeve','fabric'],['fabric', 'sleeve'],['fabric','texture']],
            'SC_UDS':[['texture', 'sleeve'],['texture', 'fabric'],['sleeve', 'texture'],['sleeve','fabric'],['fabric', 'sleeve'],['fabric','texture']],
            'LDD_UDS':[['texture', 'sleeve'],['texture', 'fabric'],['sleeve', 'texture'],['sleeve','fabric'],['fabric', 'sleeve'],['fabric','texture']],
            'SC_LDD_UDS':[['texture', 'sleeve','fabric'],['texture', 'fabric','sleeve'],['sleeve', 'texture','fabric'],['sleeve','fabric','texture'],['fabric', 'sleeve','texture'],['fabric','texture','sleeve']],
            }
        celeb_attr = {'SC':['black_hair', 'smiling', 'straight_hair'], 'LDD':['black_hair', 'smiling', 'straight_hair'],'UDS':['black_hair', 'smiling', 'straight_hair'],
            'SC_LDD':[['black_hair', 'smiling'],['black_hair', 'straight_hair'],['smiling', 'black_hair'],['smiling', 'straight_hair'],['straight_hair', 'black_hair'],['straight_hair', 'smiling']],
            'SC_UDS':[['black_hair', 'smiling'],['black_hair', 'straight_hair'],['smiling', 'black_hair'],['smiling', 'straight_hair'],['straight_hair', 'black_hair'],['straight_hair', 'smiling']],
            'LDD_UDS':[['black_hair', 'smiling'],['black_hair', 'straight_hair'],['smiling', 'black_hair'],['smiling', 'straight_hair'],['straight_hair', 'black_hair'],['straight_hair', 'smiling']],
            'SC_LDD_UDS':[['black_hair', 'smiling', 'straight_hair'],['black_hair', 'straight_hair', 'smiling'],['smiling', 'black_hair', 'straight_hair'],['smiling', 'straight_hair', 'black_hair'],['straight_hair', 'black_hair', 'smiling'],['straight_hair', 'smiling', 'black_hair']],
            }
        
        smallnorb_attr = {'SC':['azimuth', 'elevation', 'lighting'], 'LDD':['azimuth', 'elevation', 'lighting'],'UDS':['azimuth', 'elevation', 'lighting'],
            'SC_LDD':[['azimuth', 'elevation'],['azimuth', 'lighting'],['elevation', 'azimuth'],['elevation', 'lighting'],['lighting', 'azimuth'],['lighting', 'elevation']],
            'SC_UDS':[['azimuth', 'elevation'],['azimuth', 'lighting'],['elevation', 'azimuth'],['elevation', 'lighting'],['lighting', 'azimuth'],['lighting', 'elevation']],
            'LDD_UDS':[['azimuth', 'elevation'],['azimuth', 'lighting'],['elevation', 'azimuth'],['elevation', 'lighting'],['lighting', 'azimuth'],['lighting', 'elevation']],
            'SC_LDD_UDS':[['azimuth', 'elevation', 'lighting'],['azimuth', 'lighting', 'elevation'],['elevation', 'azimuth', 'lighting'],['elevation', 'lighting', 'azimuth'],['lighting', 'azimuth', 'elevation'],['lighting', 'elevation', 'azimuth']],
            }
        
        celeba_cluster = {'CLUSTER':
            {
                0: 0, 1: 1, 
                2: 2, 3: 3
            }
            }
        
        attr_dict = {'CELEBA_C': celeb_attr, 'DEEPFASHION':deep_attr,'DSPRITES':dsprites_attr, 
                    'SHAPES3D':dsprites_attr, 'SMALLNORB':smallnorb_attr,'CELEBA':celeb_attr,
                    'iwildcam':None, 'FMOW':None, 'CAMELYON17': None, 'POVERTY': None,
                    'CELEBA_CLUSTER':celeba_cluster}
        
        attr = attr_dict[dataset]
        _hparam('attr', attr, lambda r: attr)
        if dataset == 'SMALLNORB':
            _hparam('img_ch', 1, lambda r: 1)
        elif dataset == 'POVERTY':
            _hparam('img_ch', 8, lambda r: 8)
        else:
            _hparam('img_ch', 3, lambda r: 3)

    if algorithm in ['ADA','ME_ADA']:
        _hparam('loops_adv', 15, lambda r: 15)
        _hparam('k', 2, lambda r: 2)
        _hparam('epochs_min', 10, lambda r: 10)
        _hparam('gamma', 1.0, lambda r: 1.0)
        _hparam('eta', 1.0, lambda r: 1.0)
        _hparam('lr_max', 20.0, lambda r: 20.0)
        _hparam('save_img', dataset=='iwildcam' or dataset=='FMOW' or dataset=='CAMELYON17' or dataset=='POVERTY', lambda r: False)
        
    if algorithm in ['UBNet', 'UBNet_raw']:
        _hparam('base_patience', 10, lambda r: 10)
        _hparam('base_epochs', 50, lambda r: 50)

    if algorithm == 'PnD':
        _hparam('alpha1', 0.2, lambda r: 0.2)
        _hparam('alpha2', 2.0, lambda r: 2.0)
        _hparam('beta', 4.0, lambda r: 4.0)
        _hparam('loss_q', 0.7, lambda r: 0.7)
        _hparam('base_epochs', 50, lambda r: 50)
        _hparam('base_patience', 10, lambda r: 10)
        _hparam('ema_alpha', 0.7, lambda r: 0.7)
        _hparam('temperature', 0.1, lambda r: 0.1)
        #_hparam('lr2',0.0005,lambda r:0.0005)
    return hparams

def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}

def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}