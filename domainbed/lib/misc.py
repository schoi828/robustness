# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import math
import hashlib
import sys
from collections import OrderedDict
from numbers import Number
import operator
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os
import numpy as np
import torch
from collections import Counter
from itertools import cycle
import torch.nn as nn
import torch.nn.functional as F

#https://github.com/Jiaxuan-Li/PnD/tree/main
class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
     
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss

#https://github.com/Jiaxuan-Li/PnD/tree/main
class EMA:
    def __init__(self, label, num_classes=None, alpha=0.9,device = 'cuda:0'):
        self.device = device
        self.label = label.to(self.device)
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))
        self.num_classes = num_classes
        self.max = torch.zeros(self.num_classes).to(self.device)

    def update(self, data, index, curve=None, iter_range=None, step=None):
        self.parameter = self.parameter.to(self.device)
        self.updated = self.updated.to(self.device)
        index = index.to(self.device)

        if curve is None:
            self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[index] = alpha * self.parameter[index] + (1 - alpha * self.updated[index]) * data
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = torch.where(self.label == label)[0]
        return self.parameter[label_index].max()
    
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = 0

    if ver == 1:
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        loss = loss.abs_() / float(batch_size)
    elif ver == 2:
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
    else:
        raise ValueError('ver == 1 or 2')
    return loss

def conditional_mmd_rbf(source, target, label, num_class, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    loss = 0
    for i in range(num_class):
        source_i = source[label==i]
        target_i = target[label==i]
        loss += mmd_rbf(source_i, target_i)
    return loss / num_class

def club(mu, logvar, y_samples):

    sample_size = y_samples.shape[0]
    # random_index = torch.randint(sample_size, (sample_size,)).long()
    random_index = torch.randperm(sample_size).long()

    positive = - (mu - y_samples) ** 2 / logvar.exp()
    negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
    upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
    return upper_bound / 2.

def reparametrize(mu, logvar, factor=0.2):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + factor*std*eps

def loglikeli(mu, logvar, y_samples):
    return (-(mu - y_samples)**2 /logvar.exp()-logvar).mean()#.sum(dim=1).mean(dim=0)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()

def sgd(parameters, lr, weight_decay=0.0, momentum=0.0):
    opt = torch.optim.SGD(params=parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return opt

class EarlyStopper:
    
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.val_metric = -float('inf')
        self.save = None

    def early_stop(self, test_acc, save=None):
        if test_acc > self.val_metric:
            if save is not None:
                self.save = save
            self.counter = 0
            self.val_metric = test_acc
        
        elif test_acc <= (self.val_metric + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def distance(h1, h2):
    ''' distance of two networks (h1, h2 are classifiers)'''
    dist = 0.
    for param in h1.state_dict():
        h1_param, h2_param = h1.state_dict()[param], h2.state_dict()[param]
        dist += torch.norm(h1_param - h2_param) ** 2  # use Frobenius norms for matrices
    return torch.sqrt(dist)


def save_cf_matrix(cf, names, path, split='train',mode='acc', save_name=None):
    ''' 
    cf: confusion matrix
    names: names of labels (e.g. ['square','ellipse','heart'])
    path: save path
    split: ['train','val','test']
    mode: ['acc','count']
    '''
    assert mode in ['acc','count'], 'mode should be either acc or count'
    
    acc = (torch.trace(cf)/torch.sum(cf)).item()
    total_gt = torch.sum(cf, axis=1)

    if mode == 'count':
        total_p = torch.sum(cf, axis=0)
        dataframe = cf
        rows = [f'{name}={int(total_gt[idx])}' for idx, name in enumerate(names)]
        columns = [f'{name}={int(total_p[idx])}' for idx, name in enumerate(names)]
        total = f' total: {int(torch.sum(cf).item())}'
    else:
        dataframe = cf/total_gt
        rows = names
        columns = names
        total = ''

    df_cm = pd.DataFrame(dataframe, index = rows, columns = columns)
    plt.figure(figsize = (6,4))
    plt.title("{} accuracy: {:.5f}".format(split,acc)+total)
    sn.heatmap(df_cm, annot=True,fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if save_name is None:
        save_name = f'cf_matrix_{split}_{mode}.png'
    plt.savefig(os.path.join(path, save_name))

def proj(delta, adv_h, h):
    ''' return proj_{B(h, \delta)}(adv_h), Euclidean projection to Euclidean ball'''
    ''' adv_h and h are two classifiers'''
    dist = distance(adv_h, h)
    if dist <= delta:
        return adv_h
    else:
        ratio = delta / dist
        for param_h, param_adv_h in zip(h.parameters(), adv_h.parameters()):
            param_adv_h.data = param_h + ratio * (param_adv_h - param_h)
        # print("distance: ", distance(adv_h, h))
        return adv_h

def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data



def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.5f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i,j in zip(meta_train, cycle(meta_test)):
         xi, yi = minibatches[i][0], minibatches[i][1]
         xj, yj = minibatches[j][0], minibatches[j][1]

         min_n = min(len(xi), len(xj))
         pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device,cf_matrix=None):
    correct = 0
    total = 0
    weights_offset = 0
    if cf_matrix is not None:
        cf = torch.zeros(cf_matrix.num_classes,cf_matrix.num_classes)

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            
            if cf_matrix is not None:
                cf+= cf_matrix(p.argmax(1).cpu(),y.cpu())
            total += batch_weights.sum().item()
    network.train()
    if cf_matrix is None:
        return correct / total
    return correct / total, cf
class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


############################################################
# A general PyTorch implementation of KDE. Builds on:
# https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/kde.py
############################################################

class Kernel(torch.nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bw=None):
        super().__init__()
        self.bw = 0.05 if bw is None else bw

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
        return test_Xs - train_Xs

    def forward(self, test_Xs, train_Xs):
        """Computes p(x) for each x in test_Xs given train_Xs."""

    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        diffs = self._diffs(test_Xs, train_Xs)
        dims = tuple(range(len(diffs.shape))[2:])
        if dims == ():
            x_sq = diffs ** 2
        else:
            x_sq = torch.norm(diffs, p=2, dim=dims) ** 2

        var = self.bw ** 2
        exp = torch.exp(-x_sq / (2 * var))
        coef = 1. / torch.sqrt(2 * np.pi * var)

        return (coef * exp).mean(dim=1)

    def sample(self, train_Xs):
        # device = train_Xs.device
        noise = torch.randn(train_Xs.shape) * self.bw
        return train_Xs + noise

    def cdf(self, test_Xs, train_Xs):
        mus = train_Xs                                                      # kernel centred on each observation
        sigmas = torch.ones(len(mus), device=test_Xs.device) * self.bw      # bandwidth = stddev
        x_ = test_Xs.repeat(len(mus), 1).T                                  # repeat to allow broadcasting below
        return torch.mean(torch.distributions.Normal(mus, sigmas).cdf(x_))


def estimate_bandwidth(x, method="silverman"):
    x_, _ = torch.sort(x)
    n = len(x_)
    sample_std = torch.std(x_, unbiased=True)

    if method == 'silverman':
        # https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
        iqr = torch.quantile(x_, 0.75) - torch.quantile(x_, 0.25)
        bandwidth = 0.9 * torch.min(sample_std, iqr / 1.34) * n ** (-0.2)

    elif method.lower() == 'gauss-optimal':
        bandwidth = 1.06 * sample_std * (n ** -0.2)

    else:
        raise ValueError(f"Invalid method selected: {method}.")

    return bandwidth


class KernelDensityEstimator(torch.nn.Module):
    """The KernelDensityEstimator model."""

    def __init__(self, train_Xs, kernel='gaussian', bw_select='Gauss-optimal'):
        """Initializes a new KernelDensityEstimator.
        Args:
            train_Xs: The "training" data to use when estimating probabilities.
            kernel: The kernel to place on each of the train_Xs.
        """
        super().__init__()
        self.train_Xs = train_Xs
        self._n_kernels = len(self.train_Xs)

        if bw_select is not None:
            self.bw = estimate_bandwidth(self.train_Xs, bw_select)
        else:
            self.bw = None

        if kernel.lower() == 'gaussian':
            self.kernel = GaussianKernel(self.bw)
        else:
            raise NotImplementedError(f"'{kernel}' kernel not implemented.")

    @property
    def device(self):
        return self.train_Xs.device

    # TODO(eugenhotaj): This method consumes O(train_Xs * x) memory. Implement an iterative version instead.
    def forward(self, x):
        return self.kernel(x, self.train_Xs)

    def sample(self, n_samples):
        idxs = np.random.choice(range(self._n_kernels), size=n_samples)
        return self.kernel.sample(self.train_Xs[idxs])

    def cdf(self, x):
        return self.kernel.cdf(x, self.train_Xs)


############################################################
# PyTorch implementation of 1D distributions.
############################################################

EPS = 1e-16


class Distribution1D:
    def __init__(self, dist_function=None):
        """
        :param dist_function: function to instantiate the distribution (self.dist).
        :param parameters: list of parameters in the correct order for dist_function.
        """
        self.dist = None
        self.dist_function = dist_function

    @property
    def parameters(self):
        raise NotImplementedError

    def create_dist(self):
        if self.dist_function is not None:
            return self.dist_function(*self.parameters)
        else:
            raise NotImplementedError("No distribution function was specified during intialization.")

    def estimate_parameters(self, x):
        raise NotImplementedError

    def log_prob(self, x):
        return self.create_dist().log_prob(x)

    def cdf(self, x):
        return self.create_dist().cdf(x)

    def icdf(self, q):
        return self.create_dist().icdf(q)

    def sample(self, n=1):
        if self.dist is None:
            self.dist = self.create_dist()
        n_ = torch.Size([]) if n == 1 else (n,)
        return self.dist.sample(n_)

    def sample_n(self, n=10):
        return self.sample(n)


def continuous_bisect_fun_left(f, v, lo, hi, n_steps=32):
    val_range = [lo, hi]
    k = 0.5 * sum(val_range)
    for _ in range(n_steps):
        val_range[int(f(k) > v)] = k
        next_k = 0.5 * sum(val_range)
        if next_k == k:
            break
        k = next_k
    return k


class Normal(Distribution1D):
    def __init__(self, location=0, scale=1):
        self.location = location
        self.scale = scale
        super().__init__(torch.distributions.Normal)

    @property
    def parameters(self):
        return [self.location, self.scale]

    def estimate_parameters(self, x):
        mean = sum(x) / len(x)
        var = sum([(x_i - mean) ** 2 for x_i in x]) / (len(x) - 1)
        self.location = mean
        self.scale = torch.sqrt(var + EPS)

    def icdf(self, q):
        if q >= 0:
            return super().icdf(q)

        else:
            # To get q *very* close to 1 without numerical issues, we:
            # 1) Use q < 0 to represent log(y), where q = 1 - y.
            # 2) Use the inverse-normal-cdf approximation here:
            #    https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
            log_y = q
            return self.location + self.scale * math.sqrt(-2 * log_y)


class Nonparametric(Distribution1D):
    def __init__(self, use_kde=True, bw_select='Gauss-optimal'):
        self.use_kde = use_kde
        self.bw_select = bw_select
        self.bw, self.data, self.kde = None, None, None
        super().__init__()

    @property
    def parameters(self):
        return []

    def estimate_parameters(self, x):
        self.data, _ = torch.sort(x)

        if self.use_kde:
            self.kde = KernelDensityEstimator(self.data, bw_select=self.bw_select)
            self.bw = torch.ones(1, device=self.data.device) * self.kde.bw

    def icdf(self, q):
        if not self.use_kde:
            # Empirical or step CDF. Differentiable as torch.quantile uses (linear) interpolation.
            return torch.quantile(self.data, float(q))

        if q >= 0:
            # Find quantile via binary search on the KDE CDF
            lo = torch.distributions.Normal(self.data[0], self.bw[0]).icdf(q)
            hi = torch.distributions.Normal(self.data[-1], self.bw[-1]).icdf(q)
            return continuous_bisect_fun_left(self.kde.cdf, q, lo, hi)

        else:
            # To get q *very* close to 1 without numerical issues, we:
            # 1) Use q < 0 to represent log(y), where q = 1 - y.
            # 2) Use the inverse-normal-cdf approximation here:
            #    https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
            log_y = q
            v = torch.mean(self.data + self.bw * math.sqrt(-2 * log_y))
            return v


class Accuracy():
    """
    Tracks accuracy per class + group
    """

    def __init__(self, top_k=[1, 5]):
        self.top_k = top_k
        self.reset()

    def reset(self):
        self.correct_dict = {}
        self.total_dict = {}

    def update(self, pred_scores, gt_ys, class_names=None, group_names=None):
        sorted_ys = torch.argsort(pred_scores, dim=-1, descending=True)
        for ix in range(len(sorted_ys)):
            cls_name = str(gt_ys[ix]) if class_names is None else class_names[ix]
            grp_name = cls_name if group_names is None else group_names[ix]
            k_to_ys = {k: sorted_ys[ix][:k] for k in self.top_k}
            self._update_one(k_to_ys, gt_ys[ix], cls_name, 'class')
            self._update_one(k_to_ys, gt_ys[ix], grp_name, 'group')

    def _update_one(self, k_to_ys, gt_y, name, grp_type):
        name = str(name)

        if grp_type not in self.total_dict:
            self.total_dict[grp_type] = {}
            self.correct_dict[grp_type] = {}

        if name not in self.total_dict[grp_type]:
            self.total_dict[grp_type][name] = 0
            self.correct_dict[grp_type][name] = {k: 0 for k in k_to_ys}

        self.total_dict[grp_type][name] += 1

        for k in k_to_ys:
            ys = [int(y) for y in k_to_ys[k]]
            if int(gt_y) in ys:
                self.correct_dict[grp_type][name][k] += 1

    def get_per_group_accuracy(self, group_type='group', factor=1):
        assert group_type in self.total_dict
        per_group_accuracy = {}
        for group_name in self.correct_dict[group_type]:
            if group_name not in per_group_accuracy:
                per_group_accuracy[group_name] = {}
            for k in self.correct_dict[group_type][group_name]:
                per_group_accuracy[group_name][k] \
                    = self.correct_dict[group_type][group_name][k] / self.total_dict[group_type][group_name] * factor
        return per_group_accuracy

    def get_mean_per_group_accuracy(self, group_type='group', factor=1, topK=None):
        per_group_accuracy = self.get_per_group_accuracy(group_type)
        mpg = {}
        for k in self.top_k:
            total_acc, total_num = 0, 0
            for group_name in per_group_accuracy:
                total_acc += per_group_accuracy[group_name][k]
                total_num += 1
            mpg[k] = total_acc / total_num * factor
        if topK is not None:
            return mpg[topK]
        else:
            return mpg

    def get_accuracy(self, group_type='class', factor=1):
        acc_dict = {}
        for k in self.top_k:
            correct, total = 0, 0
            for grp_name in self.total_dict[group_type]:
                correct += self.correct_dict[group_type][grp_name][k]
                total += self.total_dict[group_type][grp_name]
            acc_dict[k] = correct / total * factor
        return acc_dict

    def summary(self, factor=100):
        obj = {}
        # obj['accuracy'] = self.get_accuracy('class', factor)
        acc = self.get_accuracy('class', factor)
        mpc = self.get_mean_per_group_accuracy('class', factor)
        mpg = self.get_mean_per_group_accuracy('group', factor)
        for k in acc:
            obj[f'Top {k} Acc'] = acc[k]
            # obj[f'Top {k} MPC'] = mpc[k]
            # obj[f'Top {k} MPG'] = mpg[k]
        return obj

    def detailed(self, factor=100):
        return self.summary(factor)
        # obj = {}
        # for group_type in self.total_dict:
        #     obj[group_type] = {
        #         'total': self.total_dict[group_type],
        #         'correct': self.correct_dict[group_type],
        #         'accuracy': self.get_accuracy(group_type, factor),
        #         'MPG': self.get_mean_per_group_accuracy(group_type, factor),
        #         'per_group': self.get_per_group_accuracy(group_type, factor)
        #     }
        # return obj

class GateWeightedCELoss():
    def __init__(self, gamma0=3, gamma=1, eps=1e-5, offset=0.1):
        self.gamma0 = gamma0
        self.gamma = gamma
        self.eps = eps
        self.offset = offset
        self.max_wt = 0  # stateful

    def __call__(self, exit_ix, curr_logits, prev_gates, gt_ys, unweighted_loss):
        curr_gt_proba = F.softmax(curr_logits, dim=1).gather(1, gt_ys.squeeze().view(-1, 1)).squeeze()
        if exit_ix == 0:
            assert prev_gates is None
            # bias-amp loss
            loss_wt = curr_gt_proba.detach() ** self.gamma0
        else:
            # weighted loss
            loss_wt = (1 - prev_gates.detach()) ** self.gamma
        curr_max_wt = loss_wt.max().detach()
        if curr_max_wt > self.max_wt:
            self.max_wt = curr_max_wt

        loss_wt = loss_wt / (self.max_wt + self.eps)
        return (loss_wt + self.offset) * unweighted_loss


class CAMSuppressionLoss():
    """
    KLD loss between uniform distribution and inconfident CAM cell locations (inconfident towards GT class)
    Inconfident regions are hard thresholded with mean CAM value
    """

    def __call__(self, cams, gt_ys):
        b, c, h, w = cams.shape
        cams = cams.reshape((b, c, h * w))
        gt_cams = torch.gather(cams, dim=1, index=gt_ys.squeeze().unsqueeze(dim=1).unsqueeze(dim=2)
                               .repeat(1, 1, h * w)).squeeze().reshape((b, h * w))
        gt_max, gt_min, gt_mean = torch.max(gt_cams, dim=1)[0], torch.min(gt_cams, dim=1)[0], torch.mean(gt_cams, dim=1)
        norm_gt_cams = (gt_cams - gt_min.unsqueeze(1)) / (gt_max.unsqueeze(1) - gt_min.unsqueeze(1)).detach()
        threshold = gt_mean.unsqueeze(1).repeat(1, h * w)

        # Assign weights so that the locations which have a score lower than the threshold are suppressed
        supp_wt = torch.where(gt_cams > threshold, torch.zeros_like(norm_gt_cams), torch.ones_like(norm_gt_cams))

        uniform_targets = torch.ones_like(cams) / c
        uniform_kld_loss = torch.sum(uniform_targets * (torch.log_softmax(uniform_targets, dim=1) -
                                                        torch.log_softmax(cams, dim=1)), dim=1)
        supp_loss = (supp_wt * uniform_kld_loss).mean()
        return supp_loss

def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)

class ExitGateLoss():
    """
    Trains the gate to exit if the sample was correctly predicted and if the overall accuracy is lower than the threshold
    """

    def __init__(self, acc_threshold, balance_factor=0.5):
        self.acc_threshold = acc_threshold
        self.accuracy = Accuracy()
        self.balance_factor = balance_factor
        self.item_to_correctness = {}

    def __call__(self, item_ixs, logits, gt_ys, gates, force_use=False, eps=1e-5):
        """

        :param logits:
        :param gt_ys:
        :param gates: probability of exiting predicted by the gate
        :param force_use:
        :param eps:
        :return:
        """
        pred_ys = torch.argmax(logits, dim=1)
        self.accuracy.update(logits, gt_ys, gt_ys, gt_ys)
        mpg = self.accuracy.get_mean_per_group_accuracy('class', topK=1)

        if mpg <= self.acc_threshold or force_use:
            gate_gt = (pred_ys == gt_ys.squeeze()).long().type(gates.type())
            for item_ix, is_correct in zip(item_ixs, gate_gt):
                self.item_to_correctness[int(item_ix)] = float(is_correct)
        else:
            gate_gt = torch.FloatTensor(
                [self.item_to_correctness[int(item_ix)] for item_ix in item_ixs]).type(gates.type())
        _exit_cnt, _continue_cnt = gate_gt.sum().detach(), (1 - gate_gt).sum().detach()
        # Assign balanced weights to exit vs continue preds
        _max_cnt = max(_exit_cnt, _continue_cnt)
        _exit_cnt, _continue_cnt = _exit_cnt / _max_cnt, _continue_cnt / _max_cnt
        _gate_loss_wts = torch.where(gate_gt > 0,
                                     (torch.ones_like(gate_gt) / (_exit_cnt + eps)) ** self.balance_factor,
                                     (torch.ones_like(gate_gt) / (_continue_cnt + eps)) ** self.balance_factor)

        # gate_loss = _gate_loss_wts * F.binary_cross_entropy(gates, gate_gt, reduction='none')
        gate_loss = _gate_loss_wts * F.binary_cross_entropy_with_logits(inv_sigmoid(gates), gate_gt, reduction='none')
        # gate_loss = _gate_loss_wts * F.mse_loss(gates, gate_gt, reduction='none')
        return gate_loss.mean()

    def on_epoch_start(self):
        self.accuracy = Accuracy()

    def on_save_checkpoint(self, checkpoint, exit_ix):
        checkpoint[f'item_to_correctness_{exit_ix}'] = self.item_to_correctness

    def on_load_checkpoint(self, checkpoint, exit_ix):
        self.item_to_correctness = checkpoint[f'item_to_correctness_{exit_ix}']
