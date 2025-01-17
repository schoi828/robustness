# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None
import torchvision
from domainbed import networks
from domainbed.networks import AugNet, L2D_ResNet, AvgFixedCentroids, PnDNet, OrthoNet, B_VAE, OccamResNet
from tqdm import tqdm
from domainbed.lib.misc import *
from domainbed.lib.kmeans import kmeans

import os
from torchvision import transforms

ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'Transfer',
    'CausIRL_CORAL',
    'CausIRL_MMD',
    'EQRM',
    'MLP',
    "VIT",
    'ADA',
    'ME_ADA',
    'L2D',
    'BetaVAE',
    'BPA',
    'PnD',
    'UBNet',
    'OccamNets'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError
    
    def update_cf(self, minibatches, unlabeled=None, cf_matrix=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """       

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        with torch.no_grad():
            p = self.predict(all_x)
        #correct = p.argmax(1).eq(all_y).sum().item()
        #train_acc = correct/all_x.shape[0]
        #step_vals['train_acc'] = train_acc
        if cf_matrix is not None:
            cf_train = cf_matrix(p.argmax(1).cpu(),all_y.cpu())

        step_vals = self.update(minibatches=minibatches,unlabeled=unlabeled)

        if cf_matrix is None:
            return step_vals
        return step_vals, cf_train
    
    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        self.network.train()

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    def get_feature(self,x):
        return self.featurizer(x).squeeze()

class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        input_grad = autograd.grad(
            F.cross_entropy(disc_out, disc_labels, reduction='sum'),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_meta_test = hparams['n_meta_test']

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches, self.num_meta_test):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}

class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, minibatches, unlabeled=None):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)

            env_grads = autograd.grad(env_loss, self.network.parameters())
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {'loss': mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, in_features, num_classes, num_domains, hparams):
        super(IGA, self).__init__(in_features, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None):
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grad = autograd.grad(env_loss, self.network.parameters(),
                                        create_graph=True)

            grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(mean_loss, self.network.parameters(),
                                        retain_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams['penalty'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': mean_loss.item(), 'penalty': penalty_value.item()}


class SelfReg(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SelfReg, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        input_feat_size = self.featurizer.n_outputs
        hidden_size = input_feat_size if input_feat_size==2048 else input_feat_size*2

        self.cdpl = nn.Sequential(
                            nn.Linear(input_feat_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, input_feat_size),
                            nn.BatchNorm1d(input_feat_size)
        )

    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        lam = np.random.beta(0.5, 0.5)

        batch_size = all_y.size()[0]

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex==val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y

        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)

        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end-ex)+ex
            shuffle_indices2 = torch.randperm(end-ex)+ex
            for idx in range(end-ex):
                output_2[idx+ex] = output[shuffle_indices[idx]]
                feat_2[idx+ex] = proj[shuffle_indices[idx]]
                output_3[idx+ex] = output[shuffle_indices2[idx]]
                feat_3[idx+ex] = proj[shuffle_indices2[idx]]
            ex = end

        # mixup
        output_3 = lam*output_2 + (1-lam)*output_3
        feat_3 = lam*feat_2 + (1-lam)*feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)

        cl_loss = F.cross_entropy(output, all_y)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale*(lam*(L_ind_logit + L_ind_feat)+(1-lam)*(L_hdl_logit + L_hdl_feat))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SANDMask(ERM):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    <https://arxiv.org/abs/2106.02266>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]
        self.k = hparams["k"]
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            betas=betas
        )

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):

        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)
            env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        # gradient masking applied here
        self.mask_grads(param_gradients, self.network.parameters())
        self.optimizer.step()
        self.update_count += 1

        return {'loss': mean_loss}

    def mask_grads(self, gradients, params):
        '''
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
        parameter based on the agreement of gradients coming from different environments.
        '''
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))



class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)

class TRM(Algorithm):
    """
    Learning Representations that Support Robust Transfer of Predictors
    <https://arxiv.org/abs/2110.09940>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TRM, self).__init__(input_shape, num_classes, num_domains,hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.num_domains = num_domains
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes).cuda()
        self.clist = [nn.Linear(self.featurizer.n_outputs, num_classes).cuda() for i in range(num_domains+1)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(num_domains+1)]

        self.optimizer_f = torch.optim.Adam(
            self.featurizer.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # initial weights
        self.alpha = torch.ones((num_domains, num_domains)).cuda() - torch.eye(num_domains).cuda()

    @staticmethod
    def neum(v, model, batch):
        def hvp(y, w, v):

            # First backprop
            first_grads = autograd.grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
            first_grads = torch.nn.utils.parameters_to_vector(first_grads)
            # Elementwise products
            elemwise_products = first_grads @ v
            # Second backprop
            return_grads = autograd.grad(elemwise_products, w, create_graph=True)
            return_grads = torch.nn.utils.parameters_to_vector(return_grads)
            return return_grads

        v = v.detach()
        h_estimate = v
        cnt = 0.
        model.eval()
        iter = 10
        for i in range(iter):
            model.weight.grad *= 0
            y = model(batch[0].detach())
            loss = F.cross_entropy(y, batch[1].detach())
            hv = hvp(loss, model.weight, v)
            v -= hv
            v = v.detach()
            h_estimate = v + h_estimate
            h_estimate = h_estimate.detach()
            # not converge
            if torch.max(abs(h_estimate)) > 10:
                break
            cnt += 1

        model.train()
        return h_estimate.detach()

    def update(self, minibatches, unlabeled=None):

        loss_swap = 0.0
        trm = 0.0

        if self.update_count >= self.hparams['iters']:
            # TRM
            if self.hparams['class_balanced']:
                # for stability when facing unbalanced labels across environments
                for classifier in self.clist:
                    classifier.weight.data = copy.deepcopy(self.classifier.weight.data)
            self.alpha /= self.alpha.sum(1, keepdim=True)

            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            # updating original network
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

            for i in range(30):
                all_logits_idx = 0
                loss_erm = 0.
                for j, (x, y) in enumerate(minibatches):
                    # j-th domain
                    feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature.detach()), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            # collect (feature, y)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            # estimate transfer risk
            for Q, (x, y) in enumerate(minibatches):
                sample_list = list(range(len(minibatches)))
                sample_list.remove(Q)

                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)

                loss_P = [F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i])*(self.alpha[Q, i].data.detach())
                          if i in sample_list else 0. for i in range(len(minibatches))]
                loss_P_sum = sum(loss_P)
                grad_P = autograd.grad(loss_P_sum, self.clist[Q].weight, create_graph=True)
                vec_grad_P = nn.utils.parameters_to_vector(grad_P).detach()
                vec_grad_P = self.neum(vec_grad_P, self.clist[Q], (feature_split[Q], y_split[Q]))

                loss_swap += loss_P_sum - self.hparams['cos_lambda'] * (vec_grad_P.detach() @ vec_grad_Q)

                for i in sample_list:
                    self.alpha[Q, i] *= (self.hparams["groupdro_eta"] * loss_P[i].data).exp()

            loss_swap /= len(minibatches)
            trm /= len(minibatches)
        else:
            # ERM
            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

        nll = loss.item()
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        if self.update_count >= self.hparams['iters']:
            loss_swap = (loss + loss_swap)
        else:
            loss_swap = loss

        loss_swap.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        loss_swap = loss_swap.item() - nll
        self.update_count += 1

        return {'nll': nll, 'trm_loss': loss_swap}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def train(self):
        self.featurizer.train()

    def eval(self):
        self.featurizer.eval()

class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IB_penalty': ib_penalty.item()}

class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        irm_penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        irm_penalty = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            irm_penalty += self._irm_penalty(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        irm_penalty /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += irm_penalty_weight * irm_penalty
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IRM_penalty': irm_penalty.item(),
                'IB_penalty': ib_penalty.item()}


class AbstractCAD(Algorithm):
    """Contrastive adversarial domain bottleneck (abstract class)
    from Optimal Representations for Covariate Shift <https://arxiv.org/abs/2201.00057>
    """

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, is_conditional):
        super(AbstractCAD, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        params = list(self.featurizer.parameters()) + list(self.classifier.parameters())

        # parameters for domain bottleneck loss
        self.is_conditional = is_conditional  # whether to use bottleneck conditioned on the label
        self.base_temperature = 0.07
        self.temperature = hparams['temperature']
        self.is_project = hparams['is_project']  # whether apply projection head
        self.is_normalized = hparams['is_normalized'] # whether apply normalization to representation when computing loss

        # whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
        # the two versions have the same optima, but we find the latter is more stable
        self.is_flipped = hparams["is_flipped"]

        if self.is_project:
            self.project = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 128),
            )
            params += list(self.project.parameters())

        # Optimizers
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def bn_loss(self, z, y, dom_labels):
        """Contrastive based domain bottleneck loss
         The implementation is based on the supervised contrastive loss (SupCon) introduced by
         P. Khosla, et al., in “Supervised Contrastive Learning“.
        Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
        """
        device = z.device
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = ~torch.eye(batch_size).bool().to(device)  # drop the "current"/"self" example
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (~mask_d)  # contain the same label but from different domains
        mask_y_d = mask_y & mask_d  # contain the same label and the same domain
        mask_y, mask_drop, mask_y_n_d, mask_y_d = mask_y.float(), mask_drop.float(), mask_y_n_d.float(), mask_y_d.float()

        # compute logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if not self.is_conditional:
            # unconditional CAD loss
            denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
            log_prob = logits - denominator

            mask_valid = (mask_y.sum(1) > 0)
            log_prob = log_prob[mask_valid]
            mask_d = mask_d[mask_valid]

            if self.is_flipped:  # maximize log prob of samples from different domains
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1)
            else:  # minimize log prob of samples from same domain
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1)
        else:
            # conditional CAD loss
            if self.is_flipped:
                mask_valid = (mask_y_n_d.sum(1) > 0)
            else:
                mask_valid = (mask_y_d.sum(1) > 0)

            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            logits = logits[mask_valid]

            # compute log_prob_y with the same label
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if self.is_flipped:  # maximize log prob of samples from different domains and with same label
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1)
            else:  # minimize log prob of samples from same domains and with same label
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1)

        def finite_mean(x):
            # only 1D for now
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return finite_mean(bn_loss)

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        bn_loss = self.bn_loss(all_z, all_y, all_d)
        clf_out = self.classifier(all_z)
        clf_loss = F.cross_entropy(clf_out, all_y)
        total_loss = clf_loss + self.hparams['lmbda'] * bn_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"clf_loss": clf_loss.item(), "bn_loss": bn_loss.item(), "total_loss": total_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class CAD(AbstractCAD):
    """Contrastive Adversarial Domain (CAD) bottleneck

       Properties:
       - Minimize I(D;Z)
       - Require access to domain labels but not task labels
       """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=False)


class CondCAD(AbstractCAD):
    """Conditional Contrastive Adversarial Domain (CAD) bottleneck

    Properties:
    - Minimize I(D;Z|Y)
    - Require access to both domain labels and task labels
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CondCAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=True)


class Transfer(Algorithm):
    '''Algorithm 1 in Quantifying and Improving Transferability in Domain Generalization (https://arxiv.org/abs/2106.03632)'''
    ''' tries to ensure transferability among source domains, and thus transferabiilty between source and target'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Transfer, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.d_steps_per_g = hparams['d_steps_per_g']

        # Architecture
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier.load_state_dict(self.classifier.state_dict())

        # Optimizers
        if self.hparams['gda']:
            self.optimizer = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr'])
        else:
            self.optimizer = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.adv_opt = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr_d'])

    def loss_gap(self, minibatches, device):
        ''' compute gap = max_i loss_i(h) - min_j loss_j(h), return i, j, and the gap for a single batch'''
        max_env_loss, min_env_loss =  torch.tensor([-float('inf')], device=device), torch.tensor([float('inf')], device=device)
        for x, y in minibatches:
            p = self.adv_classifier(self.featurizer(x))
            loss = F.cross_entropy(p, y)
            if loss > max_env_loss:
                max_env_loss = loss
            if loss < min_env_loss:
                min_env_loss = loss
        return max_env_loss - min_env_loss

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        # outer loop
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del all_x, all_y
        gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
        self.optimizer.zero_grad()
        gap.backward()
        self.optimizer.step()
        self.adv_classifier.load_state_dict(self.classifier.state_dict())
        for _ in range(self.d_steps_per_g):
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
        return {'loss': loss.item(), 'gap': -gap.item()}

    def update_second(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count = (self.update_count + 1) % (1 + self.d_steps_per_g)
        if self.update_count.item() == 1:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del all_x, all_y
            gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            self.optimizer.zero_grad()
            gap.backward()
            self.optimizer.step()
            self.adv_classifier.load_state_dict(self.classifier.state_dict())
            return {'loss': loss.item(), 'gap': gap.item()}
        else:
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
            return {'gap': -gap.item()}


    def predict(self, x):
        return self.classifier(self.featurizer(x))


class AbstractCausIRL(ERM):
    '''Abstract class for Causality based invariant representation learning algorithm from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractCausIRL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        first = None
        second = None

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i] + 1e-16, targets[i])
            slice = np.random.randint(0, len(features[i]))
            if first is None:
                first = features[i][:slice]
                second = features[i][slice:]
            else:
                first = torch.cat((first, features[i][:slice]), 0)
                second = torch.cat((second, features[i][slice:]), 0)
        if len(first) > 1 and len(second) > 1:
            penalty = torch.nan_to_num(self.mmd(first, second))
        else:
            penalty = torch.tensor(0)
        objective /= nmb

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class CausIRL_MMD(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the MMD distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_MMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=True)


class CausIRL_CORAL(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the CORAL distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_CORAL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=False)


class EQRM(ERM):
    """
    Empirical Quantile Risk Minimization (EQRM).
    Algorithm 1 from [https://arxiv.org/pdf/2207.09944.pdf].
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, dist=None):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.register_buffer('alpha', torch.tensor(self.hparams["eqrm_quantile"], dtype=torch.float64))
        if dist is None:
            self.dist = Nonparametric()
        else:
            self.dist = dist

    def risk(self, x, y):
        return F.cross_entropy(self.network(x), y).reshape(1)

    def update(self, minibatches, unlabeled=None):
        env_risks = torch.cat([self.risk(x, y) for x, y in minibatches])

        if self.update_count < self.hparams["eqrm_burnin_iters"]:
            # Burn-in/annealing period uses ERM like penalty methods (which set penalty_weight=0, e.g. IRM, VREx.)
            loss = torch.mean(env_risks)
        else:
            # Loss is the alpha-quantile value
            self.dist.estimate_parameters(env_risks)
            loss = self.dist.icdf(self.alpha)

        if self.update_count == self.hparams['eqrm_burnin_iters']:
            # Reset Adam (like IRM, VREx, etc.), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["eqrm_lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        return {'loss': loss.item()}

class MLP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.network = networks.MLP(np.product(input_shape), num_classes, self.hparams)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        return self.network(x)

class VIT(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # if hparams['pretrain']:
        #     self.network = torchvision.models.vit_b_16(weights=True)
        #     self.network.heads = nn.Linear(768,num_classes)
        # else:
        #     self.network = VisionTransformer(
        #         image_size=input_shape[-1],
        #         in_ch=self.hparams['img_ch'],
        #         num_classes=num_classes,
        #         patch_size=self.hparams['patch_size'],
        #         hidden_dim=256,
        #         mlp_dim=512,
        #         num_heads=8,
        #         num_layers=8,
        #         dropout=0.1,
        #     )
        self.network = torchvision.models.vit_b_16(weights=hparams['pretrain'])
        self.network.heads = nn.Linear(768,num_classes)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x: torch.Tensor):
        # x = x.view(x.shape[0], -1)
        return self.network(x)

    def get_feature(self, x):
        return self.network(x).squeeze()

class Denormalise(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-12)
        mean_inv = -mean * std_inv
        super(Denormalise, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(Denormalise, self).__call__(tensor.clone())


class ADA(ERM):
    """
    https://github.com/garyzhao/ME-ADA
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.dist_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.output_dir = None
        self.counter_k = 0
        self.epoch = 0
        self.maximized_epochs = set()
        self.image_denormalise = Denormalise([0.5] * hparams['img_ch'], [0.5] * hparams['img_ch'])
        if hparams['save_img']:
            self.to_pil = transforms.ToPILImage()
            self.img_id = 0

    def init_dataloader(self,shuffle=True):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.hparams['batch_size'],
            shuffle=shuffle,
            num_workers=self.train_data.N_WORKERS,
            pin_memory=True)

    def init_iter_loader(self):
        self.iter_loader = iter(self.train_loader)

    def max_loss(self, preds, targets, outputs_embedding, inputs_embedding):
        return self.loss_fn(preds, targets) - self.hparams['gamma'] * self.dist_fn(
                    outputs_embedding, inputs_embedding)
    
    def maximize(self):
        self.network.eval()

        #self.train_data.transform = self.preprocess
        self.init_dataloader()
        images, labels = [], []
        for i, (images_train, labels_train) in enumerate(self.train_loader):

            # wrap the inputs and labels in Variable
            inputs, targets = images_train.cuda(), labels_train.cuda()

            # forward with the adapted parameters
            inputs_embedding = self.network[0](inputs).detach().clone()
            inputs_embedding.requires_grad_(False)

            inputs_max = inputs.detach().clone()
            inputs_max.requires_grad_(True)
            optimizer = sgd(parameters=[inputs_max], lr=self.hparams['lr_max'])

            for ite_max in range(self.hparams['loops_adv']):

                outputs_embedding = self.network[0](inputs_max)
                preds = self.network[1](outputs_embedding)
                # loss
                loss = self.max_loss(preds, targets, outputs_embedding, inputs_embedding)

                # init the grad to zeros first
                self.network.zero_grad()
                optimizer.zero_grad()

                # backward your network
                (-loss).backward()

                # optimize the parameters
                optimizer.step()

                flags_log = os.path.join(self.output_dir, 'max_loss_log.txt')
                write_log('ite_adv:{}, {}'.format(ite_max, loss.item()), flags_log)

            inputs_max = inputs_max.detach().clone().cpu()
            for j in range(len(inputs_max)):
                input_max = self.image_denormalise(inputs_max[j])
                if self.hparams['img_ch'] == 3:
                    input_max = input_max.permute(1,2,0).clamp(min=0.0, max=1.0).numpy()
                else:
                    input_max = torch.squeeze(input_max).clamp(min=0.0, max=1.0).numpy()

                if self.hparams['save_img']:
                    img_path = os.path.join(self.train_data.ada_root,f'{self.img_id}.png')
                    images.append(img_path)
                    pil_img = self.to_pil(input_max)
                    pil_img.save(img_path)
                    self.img_id+=1
                else:
                    images.append(input_max)
                labels.append(labels_train[j].item())

        print('Maximization Done')
        if self.hparams['save_img']:
            return images, labels
        return np.stack(images), np.stack(labels)

    def update_cf(self, minibatches, unlabeled=None, cf_matrix=None):

        if ((self.epoch + 1) % self.hparams['epochs_min'] == 0) \
            and self.epoch not in self.maximized_epochs \
            and (self.counter_k < self.hparams['k']):  # if T_min iterations are passed
            print('Generating adversarial images [iter {}]'.format(self.counter_k))
            images, labels = self.maximize()
            if self.hparams['save_img']:
                self.train_data.ada_samples = [(os.path.join(self.train_data.ada_root, i), l) for (i, l) in zip(images, labels)]
                self.train_data.samples+=self.train_data.ada_samples
            else:
                self.train_data._imgs = np.concatenate([self.train_data._imgs, images])
                self.train_data._labels = np.concatenate([self.train_data._labels, labels])
            print(f'New dataset size: {len(self.train_data)}')
            self.counter_k += 1
            self.maximized_epochs.add(self.epoch)
            #update steps per epoch
            self.init_dataloader()
            self.init_iter_loader()
        self.network.train()

        #self.scheduler.T_max = counter_ite + len(self.train_loader) * (flags.epochs - epoch)
        try:
            images_train, labels_train = next(self.iter_loader)
        except:
            self.init_iter_loader()            
            images_train, labels_train = next(self.iter_loader)
            self.epoch+=1
            # wrap the inputs and labels in Variable
        inputs, labels = images_train.cuda(), labels_train.cuda()

        # forward with the adapted parameters
        outputs= self.predict(inputs)

        # loss
        loss = self.loss_fn(outputs, labels)

        # init the grad to zeros first
        self.optimizer.zero_grad()

        # backward your network
        loss.backward()

        # optimize the parameters
        self.optimizer.step()

        step_vals = {'loss':loss.item()}
        if cf_matrix is None:
            return step_vals

        cf_train = cf_matrix(outputs.argmax(1).cpu(),labels.cpu())

        return step_vals, cf_train


class ME_ADA(ADA):

    """
    https://github.com/garyzhao/ME-ADA
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
    
    def entropy_loss(self, x):
        out = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        out = -1.0 * out.sum(dim=1)
        return out.mean()
    
    def max_loss(self, preds, targets, outputs_embedding, inputs_embedding):
        return self.loss_fn(preds, targets) + self.hparams['eta'] * self.entropy_loss(preds) - \
                       self.hparams['gamma'] * self.dist_fn(outputs_embedding, inputs_embedding)

# modified from https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
class BetaVAE(MLP):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self, input_shape, num_classes, num_domains, hparams) -> None:
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.latent_dim = 256

        self.vae = B_VAE(input_shape)

        self.optimizer_vae = torch.optim.Adam(
            self.vae.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        ## define MLP
        self.network = networks.MLP(self.latent_dim, num_classes, self.hparams)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # update beta-vae
        #vae_losses = self.update_vae(all_x)
        # update MLP
        
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            z = self.vae.forward_latent(x)
        return self.network(z.detach())
    
    def init_dataloader(self, train_data, val_data):

        self.loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=train_data.N_WORKERS,
            pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=train_data.N_WORKERS,
            pin_memory=True)
    
    def save_vae(self):
        torch.save({
                'state_dict': self.vae.state_dict(), 
            }, os.path.join(self.output_dir, 'vae.pkl'))
        
    def init_centroids(self):
        pass

    def train_base_model(self):
        print('Training the base model...')
        self.best_val_loss = float('inf')
        earlystopper = EarlyStopper(patience=self.hparams['base_patience'])

        for epoch in range(self.hparams['base_epochs']):
            
            self.vae.train()
            for data, _ in self.loader:#in tqdm(self.loader, desc='Training the base model', ncols=35):
                data = data.cuda()
                out = self.vae.update_vae(data)
                vae_loss = out['vae_loss']

                self.optimizer_vae.zero_grad()
                vae_loss.backward()
                self.optimizer_vae.step()

            self.vae.eval()
            val_loss = 0
            with torch.no_grad():
                for data, _ in self.val_loader:#, desc='Running validation', ncols=35):
                    data = data.cuda()
                    out = self.vae.update_vae(data)
                    val_loss += out['vae_loss'].item()
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_vae()

            print('Base model training... Epoch: {}/{}, Current Val Loss: {:.4f}, Best Loss: {:.4f}'.format(
                epoch,self.hparams['base_epochs'], val_loss, self.best_val_loss))
            
            if earlystopper.early_stop(-val_loss):
                break
            
        val_checkpoint = torch.load(os.path.join(self.output_dir, 'vae.pkl'))
        self.vae.load_state_dict(val_checkpoint['state_dict'])
        os.remove(os.path.join(self.output_dir, 'vae.pkl'))
        
class BPA(ERM):
    """
    https://github.com/skynbe/pseudo-attributes
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        # Caffe Alexnet for singleDG task, Leave-one-out PACS DG task.
        # self.extractor = caffenet(args.n_classes).to(device)
        self.model = self.network#L2D_ResNet(self.featurizer, self.classifier)
        del self.network
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        #hparams['k'] = num_classes
        self.num_clusters = int(hparams['k'])
        self.num_classes = num_classes
        self.centroids = AvgFixedCentroids(hparams, num_classes, per_clusters=self.num_clusters)
        self.update_cluster_iter = hparams['update_cluster_iter']
        #self.checkpoint_dir = args.checkpoint_dir
        self.class_weight = None
        self.base_model = copy.deepcopy(self.model)
        self.base_optimizer = torch.optim.Adam(
            self.base_model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.i = 0
        self.output_dir = None

    def init_centroids(self):
        if not self.centroids.initialized:
            cluster_assigns, cluster_centers = self.inital_clustering()
            self.centroids.initialize_(cluster_assigns, cluster_centers)

    def init_dataloader(self, train_data, val_data):
        self.loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=train_data.N_WORKERS,
            pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=train_data.N_WORKERS,
            pin_memory=True)
    
    def save_base_model(self):
        torch.save({
                'state_dict': self.base_model.state_dict(), 
            }, os.path.join(self.output_dir, 'base_model.pkl'))
        
    def train_base_model(self):
        print('Training the base model...')
        self.best_val_acc = 0
        earlystopper = EarlyStopper(patience=self.hparams['base_patience'])

        for epoch in range(self.hparams['base_epochs']):
            
            self.base_model.train()
            for data, target, index in self.loader:#in tqdm(self.loader, desc='Training the base model', ncols=35):
                data, target, index = data.cuda(), target.cuda(), index.cuda()
                out = self.base_model(data)
                loss = torch.mean(self.criterion(out,target.long()))
                
                self.base_optimizer.zero_grad()
                loss.backward()
                self.base_optimizer.step()
            
            self.base_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in self.val_loader:#, desc='Running validation', ncols=35):
                    data, target = data.cuda(), target.cuda()
                    out = self.base_model(data)
                    B = target.size(0)
                    correct += (out.argmax(1).eq(target).float()).sum().item()
                    total+=B
            val_acc = correct/total

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_base_model()

            print('Base model training... Epoch: {}/{}, Current Val Acc: {:.4f}, Best Acc: {:.4f}'.format(
                epoch,self.hparams['base_epochs'], val_acc, self.best_val_acc))
            
            if earlystopper.early_stop(val_acc) or val_acc == 1.0:
                break
            
        val_checkpoint = torch.load(os.path.join(self.output_dir, 'base_model.pkl'))
        self.base_model.load_state_dict(val_checkpoint['state_dict'])
        os.remove(os.path.join(self.output_dir, 'base_model.pkl'))
    def _extract_features(self, model, data_loader):
        features, targets = [], []
        ids = []

        for data, target, index in tqdm(data_loader, desc='Feature extraction for clustering..', ncols=50):
            data, target, index = data.cuda(), target.cuda(), index.cuda()
            feature = model[0](data)
            out = model[1](feature)
            features.append(feature)
            targets.append(target)
            ids.append(index)

        features = torch.cat(features)
        targets = torch.cat(targets)
        ids = torch.cat(ids)
        return features, targets, ids
    
    def _cluster_features(self, data_loader, features, targets, ids, num_clusters):
        
        N = len(data_loader.dataset)
        num_classes = self.num_classes
        sorted_target_clusters = torch.zeros(N).long().cuda() + num_clusters*num_classes

        target_clusters = torch.zeros_like(targets)-1
        cluster_centers = []

        for t in range(num_classes):
            target_assigns = (targets==t).nonzero().squeeze()
            if len(target_assigns.shape) == 0:
                target_assigns = target_assigns.unsqueeze(0)
            print('target assigns', t, target_assigns.shape)
            feature_assigns = features[target_assigns]
            print('feature assigns', feature_assigns.shape)
            if len(feature_assigns.shape) == 1:
                feature_assigns = feature_assigns.unsqueeze(0).repeat(self.num_clusters,1)
            while feature_assigns.shape[0] < self.num_clusters:
                feature_assigns = torch.cat([feature_assigns,feature_assigns] , dim=0)
            cluster_ids, cluster_center = kmeans(X=feature_assigns, num_clusters=num_clusters, distance='cosine', device=0)
            cluster_ids_ = cluster_ids + t*num_clusters
            try:
                target_clusters[target_assigns] = cluster_ids_.cuda()
            except:
                target_clusters[target_assigns] = cluster_ids_.cuda()[:target_assigns.shape[0]]
                print('target clusters', target_clusters[target_assigns].shape)
                print('target assigns', target_assigns.shape)
                print('cluster ids', cluster_ids_.shape)
            if len(cluster_center.shape) == 1:
                print('removed cluster center', t, cluster_center.shape)
                continue
                
                #cluster_center = cluster_center.unsqueeze(0)
            print('added cluster_center', t, cluster_center.shape)
            cluster_centers.append(cluster_center)

        sorted_target_clusters[ids] = target_clusters
        cluster_centers = torch.cat(cluster_centers, 0)
        return sorted_target_clusters, cluster_centers

    def inital_clustering(self):
        data_loader = self.loader
        self.base_model.eval()
        
        with torch.no_grad():

            features, targets, ids = self._extract_features(self.base_model, data_loader)
            num_clusters = self.num_clusters
            cluster_assigns, cluster_centers = self._cluster_features(data_loader, features, targets, ids, num_clusters)
            
            cluster_counts = cluster_assigns.bincount().float()
            print("Cluster counts : {}, len({})".format(cluster_counts, len(cluster_counts)))
    
        return cluster_assigns, cluster_centers
    
    def predict(self, x):
        return self.model(x)
    
    def update_cf(self, minibatches, unlabeled=None, cf_matrix=None):
        self.i+=1
        data = torch.cat([x for x, _,_ in minibatches])
        target = torch.cat([y for _, y,_ in minibatches])

        _,_ ,ids = minibatches[0]

        #results = self.model(data)
        results = {}
        feature = self.model[0](data)
        out = self.model[1](feature)
        results['feature'] = feature
        results['out'] = out
        
        weight = self.centroids.get_cluster_weights(ids)
        if torch.isnan(weight)[0]:
            print('nan')
        weight =  torch.nan_to_num(weight)
        loss = torch.mean(self.criterion(results["out"], target.long()) * (weight))

        self.optimizer.zero_grad()
        (loss).backward()
        self.optimizer.step()
        if torch.isnan(loss):
            print(self.i, weight)
            print('data', torch.max(data),torch.min(data))
           
            #print('logits', self.convertor.shift_var,self.convertor.shift_mean )
            raise Exception
        if self.centroids.initialized:
            self.centroids.update(results, target, ids)
            if self.update_cluster_iter > 0 and self.i % self.update_cluster_iter == 0:
                self.centroids.compute_centroids()
        step_vals = {'loss':loss.item()}
        if cf_matrix is None:
            return step_vals
        cf_train = cf_matrix(out.argmax(1).cpu(),target.cpu())
        return step_vals, cf_train

class L2D(ERM):
    """
    https://github.com/zcaicaros/L2D
    """    
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.counterk=0

        # Caffe Alexnet for singleDG task, Leave-one-out PACS DG task.
        # self.extractor = caffenet(args.n_classes).to(device)
        self.extractor = L2D_ResNet(self.featurizer, num_classes)
        del self.network
        self.convertor = AugNet(1,img_shape=input_shape)
        
        self.optimizer = torch.optim.Adam(
            self.extractor.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.convertor_opt = torch.optim.SGD(self.convertor.parameters(), lr=hparams['lr_sc'])
        
        self.n_classes = num_classes
        self.centroids = 0
        self.d_representation = 0
        self.flag = False
        self.con = SupConLoss()
        self.tran = transforms.Normalize([0.5] * 3, [0.5] * 3) #use 3 channels regardless of input img
        self.alpha2 = self.hparams['alpha2']
        self.alpha1 = self.hparams['alpha1']
        self.beta = self.hparams['beta']

    def predict(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        return self.extractor(x, train=False)[0]
    
    def update_cf(self, minibatches, unlabeled=None, cf_matrix=None):
        data = torch.cat([x for x, y in minibatches])
        class_l = torch.cat([y for x, y in minibatches])
        
        # Stage 1
        self.optimizer.zero_grad()

        if data.shape[1] == 1:
            data = data.repeat(1,3,1,1)
        # Aug        
        inputs_max = self.tran(torch.sigmoid(self.convertor(data)))
        inputs_max = inputs_max * 0.6 + data * 0.4
        data_aug = torch.cat([inputs_max, data])
        labels = torch.cat([class_l, class_l])

        # forward
        logits, tuple = self.extractor(data_aug)

        # Maximize MI between z and z_hat
        emb_src = F.normalize(tuple['Embedding'][:class_l.size(0)]).unsqueeze(1)
        emb_aug = F.normalize(tuple['Embedding'][class_l.size(0):]).unsqueeze(1)
        con = self.con(torch.cat([emb_src, emb_aug], dim=1), class_l)

        # Likelihood
        mu = tuple['mu'][class_l.size(0):]
        logvar = tuple['logvar'][class_l.size(0):]
        y_samples = tuple['Embedding'][:class_l.size(0)]
        likeli = -loglikeli(mu, logvar, y_samples)

        # Total loss & backward
        class_loss = F.cross_entropy(logits, labels)

        loss = class_loss + self.alpha2*likeli + self.alpha1*con
        loss.backward()
        self.optimizer.step()
        _, cls_pred = logits.max(dim=1)

        # STAGE 2
        inputs_max =self.tran(torch.sigmoid(self.convertor(data, estimation=True)))
        inputs_max = inputs_max * 0.6 + data * 0.4
        data_aug = torch.cat([inputs_max, data])

        # forward with the adapted parameters
        outputs, tuples = self.extractor(x=data_aug)

        # Upper bound MI
        mu = tuples['mu'][class_l.size(0):]
        logvar = tuples['logvar'][class_l.size(0):]
        y_samples = tuples['Embedding'][:class_l.size(0)]
        div = club(mu, logvar, y_samples)
        # div = criterion(outputs, labels)
        
        # Semantic consistency
        e = tuples['Embedding']
        e1 = e[:class_l.size(0)]
        e2 = e[class_l.size(0):]
        dist = conditional_mmd_rbf(e1, e2, class_l, num_class=self.n_classes)

        if torch.isnan(class_loss):
            class_loss = torch.nan_to_num(class_loss)
        if torch.isnan(dist):
            dist = torch.nan_to_num(dist)
            print('dist nan')
        if torch.isnan(div):
            div = torch.nan_to_num(div)
            print('div nan')
            #print(class_loss, dist, div)
            #print('data', torch.max(data),torch.min(data))
            #print('e1',e1)
            #print('e2',e2)
            #print('mu',mu)
            #print('logvar',logvar)
            #print('logits', self.convertor.shift_var,self.convertor.shift_mean )
            #raise Exception
        # Total loss and backward
        self.convertor_opt.zero_grad()
        (dist + self.beta * div).backward()
        self.convertor_opt.step()
      
        step_vals = {"class_loss": class_loss.item(),
                    "AUG_acc": torch.sum(cls_pred[:class_l.size(0)] == class_l.data).item() / class_l.shape[0],
                    "RAW_acc": torch.sum(cls_pred[class_l.size(0):] == class_l.data).item() / class_l.shape[0]}

        if cf_matrix is None:
            return step_vals
        
        cf_train = cf_matrix(logits[class_l.size(0):].argmax(1).cpu(),class_l.cpu())
        return step_vals, cf_train


class PnD(Algorithm):
    """
    https://github.com/Jiaxuan-Li/PnD
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.alpha1 = self.hparams['alpha1']
        self.alpha2 = self.hparams['alpha2']
        self.beta = self.hparams['beta']
        self.best_valid_acc = 0
        self.intri_criterion = nn.CrossEntropyLoss(reduction='none')
        self.bias_criterion = GeneralizedCELoss(q=hparams['loss_q'])
        self.temperature = hparams['temperature']
        self.model = PnDNet(self.input_shape, self.num_classes,pretrained=self.hparams['pretrain'],hparams=hparams).cuda()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    def init_dataloader(self, train_data, val_data):
        train_target_attr = []
        print('init dataloader')
        try:
            train_target_attr = train_data.dataset.y_array
        except:
            try:
                for i in train_data._labels:     
                    train_target_attr.append(int(i))
            except:
                for i in range(len(train_data)):
                    _, label,_ = train_data[i]     
                    train_target_attr.append(int(label))
        print('init dataloader', len(train_target_attr))
        train_target_attr = torch.LongTensor(train_target_attr)  
        self.sample_loss_ema_b0 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=self.hparams['ema_alpha'])
        self.sample_loss_ema_d0 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=self.hparams['ema_alpha'])
        self.sample_loss_ema_b1 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=self.hparams['ema_alpha'])
        self.sample_loss_ema_d1 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=self.hparams['ema_alpha'])
        self.sample_loss_ema_b2 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=self.hparams['ema_alpha'])
        self.sample_loss_ema_d2 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=self.hparams['ema_alpha'])
        self.sample_loss_ema_b3 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=self.hparams['ema_alpha'])
        self.sample_loss_ema_d3 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=self.hparams['ema_alpha'])

        self.loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=train_data.N_WORKERS,
            pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=train_data.N_WORKERS,
            pin_memory=True)
    
    def save_base_model(self):
        torch.save({
                'state_dict': self.model.state_dict(), 
            }, os.path.join(self.output_dir, 'base_model.pkl'))

    def kl_loss(self, x_pred, x_gt):
        kl_gt = F.softmax(x_gt, dim=-1)
        kl_pred = F.log_softmax(x_pred, dim=-1)
        tmp_loss = F.kl_div(kl_pred, kl_gt, reduction='none')
        tmp_loss = torch.exp(-tmp_loss).mean()
        return tmp_loss
    
    def contrastive_loss(self, y_mix, y_ori, indices_mini):
        temperature = self.temperature
        y_pos, y_neg = y_mix
        y_pos = y_pos.unsqueeze(1)
        y_neg = y_neg.unsqueeze(1)
        bs = int(y_ori.size(0)/16)
        y_neg = torch.reshape(y_neg, (y_pos.size(0), bs, y_pos.size(2)))
        y_ori = y_ori[indices_mini]
        y_ori = y_ori.unsqueeze(1)
        y_all = torch.cat([y_pos, y_neg], dim=1)
        y_expand = y_ori.repeat(1, 9, 1)     
        neg_dist = -((y_expand - y_all) ** 2).mean((2)) * temperature
        label = torch.zeros(16).long().cuda()
        contrastive_loss_euclidean = nn.CrossEntropyLoss()(neg_dist, label)
        return contrastive_loss_euclidean
    
    def board_ours_acc(self, step):
        # check label network
        valid_accs, valid_accs_cls, valid_loss = self.evaluate_ours(data_loader = self.val_loader) 
        if valid_accs >= self.best_valid_acc:
            self.best_valid_acc = valid_accs
            self.save_base_model()
        print(f'valid acc: {valid_accs}')
        print(f'valid loss : {valid_loss}')
        print(f'valid accs cls: {valid_accs_cls}')
        print(f'BEST valid acc: {self.best_valid_acc}')
        return valid_accs
        #test_accs, test_accs_cls, test_loss = self.evaluate_ours(data_loader = self.test_loader) 
        #print(f'test acc: {test_accs}')
        #print(f'test loss: {test_loss}')
        #print(f'test accs cls: {test_accs_cls}')

   # evaluation code for ours
    def evaluate_ours(self, data_loader):
        self.model.eval()
        total_correct, total_num = 0, 0
        total_correct_cls = np.zeros(self.num_classes)
        total_num_cls = np.zeros(self.num_classes)
        loss_all = []
        for data, label in tqdm(data_loader, leave=False, desc= 'Evaluation'):
            data = data.cuda()
            label = label.cuda()     
            with torch.no_grad():
                _, logits_gate  = self.model(data)
                pred_label = logits_gate['dm_conflict_out']
                loss_i = self.intri_criterion(pred_label, label)
                loss_all.append(loss_i.mean().item())
                pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]
                for i in range(self.num_classes):
                    correct_i = (pred[label==i] == label[label==i]).long()
                    total_correct_cls[i] += correct_i.sum()
                    total_num_cls[i] += correct_i.shape[0]

        loss_all = np.mean(loss_all)
        accs = total_correct/float(total_num)
        accs_cls = np.divide(total_correct_cls,total_num_cls)
        return accs,accs_cls,loss_all
    
    def train_base_model(self):
        print('Training the base model...')
        self.best_val_acc = 0
        earlystopper = EarlyStopper(patience=self.hparams['base_patience'])

        for epoch in range(self.hparams['base_epochs']):
            
            self.model.train()
            for data, label, index in self.loader:#in tqdm(self.loader, desc='Training the base model', ncols=35):
                self.optimizer.zero_grad()   
                data, label, index = data.cuda(), label.cuda(), index.cuda()
                logits_all, logits_gate = self.model(data, y = label, use_mix = False)  
                
                loss_dis_conflict_all = torch.zeros((4,data.size(0))).float()
                loss_dis_align_all = torch.zeros((4,data.size(0))).float()

                for i in range(4):
                    key_conflict_i = f"E={i}, dm_conflict_out"
                    key_align_i = f"E={i}, dm_align_out"   
                    pred_conflict_i = logits_all[key_conflict_i]
                    pred_align_i = logits_all[key_align_i]    
                    loss_dis_conflict_i_ = self.intri_criterion(pred_conflict_i, label).detach()
                    loss_dis_align_i_ = self.intri_criterion(pred_align_i, label).detach()

                    # EMA sample loss
                    getattr(self, f'sample_loss_ema_d{i}').update(loss_dis_conflict_i_, index)
                    getattr(self, f'sample_loss_ema_b{i}').update(loss_dis_align_i_, index)

                    # class-wise normalize
                    loss_dis_conflict_i_ = getattr(self, f'sample_loss_ema_d{i}').parameter[index].clone().detach()
                    loss_dis_align_i_ = getattr(self, f'sample_loss_ema_b{i}').parameter[index].clone().detach()

                    loss_dis_conflict_i_ = loss_dis_conflict_i_.cuda()
                    loss_dis_align_i_ = loss_dis_align_i_.cuda()

                    for c in range(self.num_classes):
                        class_index = torch.where(label == c)[0].cuda()
                        max_loss_conflict = getattr(self, f'sample_loss_ema_d{i}').max_loss(c)
                        max_loss_align = getattr(self, f'sample_loss_ema_b{i}').max_loss(c)
                        loss_dis_conflict_i_[class_index] /= max_loss_conflict
                        loss_dis_align_i_[class_index] /= max_loss_align

                    loss_weight_i  = loss_dis_align_i_ / (loss_dis_align_i_+ loss_dis_conflict_i_ + 1e-8)     
                    loss_dis_conflict_i = self.intri_criterion(pred_conflict_i, label) * loss_weight_i.cuda() 
                    #print(pred_align_i,pred_align_i.shape)
                    loss_dis_align_i = self.bias_criterion(pred_align_i, label)                                            
                    loss_dis_conflict_all[i,:] = loss_dis_conflict_i
                    loss_dis_align_all[i,:] = loss_dis_align_i         
                loss_dis_experts  = self.alpha1*loss_dis_conflict_all.mean() +  loss_dis_align_all.mean()
                
                kl_loss_conflict_1 = self.kl_loss(logits_all['E=1, dm_align_out'], logits_all['E=0, dm_align_out'].detach())
                kl_loss_conflict_2 = self.kl_loss(logits_all['E=2, dm_align_out'], logits_all['E=1, dm_align_out'].detach())         
                kl_loss_conflict_3 = self.kl_loss(logits_all['E=3, dm_align_out'], logits_all['E=2, dm_align_out'].detach())  
                kl_loss_conflict = kl_loss_conflict_1 + kl_loss_conflict_2 + kl_loss_conflict_3
                            
                loss_experts = 4*loss_dis_experts + kl_loss_conflict     
           
                ######gate
                pred_conflict = logits_gate['dm_conflict_out']
                loss_dis_conflict = self.intri_criterion(pred_conflict, label)          
                loss_gate  = loss_dis_conflict.mean()                

                loss = loss_experts + loss_gate           
                loss.backward()
                self.optimizer.step()   

            print('Base model training... Epoch: {}/{}'.format(epoch,self.hparams['base_epochs']))
            val_acc = self.board_ours_acc(epoch)
            if earlystopper.early_stop(val_acc) or val_acc == 1.0:
                break
            #break####################################################################################################################
        val_checkpoint = torch.load(os.path.join(self.output_dir, 'base_model.pkl'))
        self.model.load_state_dict(val_checkpoint['state_dict'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        os.remove(os.path.join(self.output_dir, 'base_model.pkl'))

    def init_centroids(self):
        pass

    def predict(self, x):
        _, logits_gate  = self.model(x)
        pred_label = logits_gate['dm_conflict_out']
        return pred_label
    
    def get_feature(self, x):
        feature_dict = self.model(x, feature=True)
        return F.adaptive_avg_pool2d(feature_dict[3],1).squeeze()
    
    def update_cf(self, minibatches, unlabeled=None, cf_matrix=None):
        self.optimizer.zero_grad()   
        data = torch.cat([x for x, _,_ in minibatches])
        label = torch.cat([y for _, y,_ in minibatches])

        bs_diff = 128 - data.size(0)

        _,_ ,index = minibatches[0]
        if bs_diff >0:
            while data.size(0) < bs_diff:
                data = torch.cat([data,data],dim=0)
                label = torch.cat([label,label],dim=0)
                index = torch.cat([index,index],dim=0)
                bs_diff = 128 - data.size(0)
            data = torch.cat([data,data[:bs_diff]],dim=0)
            label = torch.cat([label,label[:bs_diff]],dim=0)
            index = torch.cat([index,index[:bs_diff]],dim=0)
        #results = self.model(data)
        logits_all, logits_gate = self.model(data, y = label, use_mix = True)

        loss_dis_conflict_all = torch.zeros((4,data.size(0))).float()
        loss_dis_align_all = torch.zeros((4,data.size(0))).float()
        loss_cf = torch.zeros((4,16)).float()
        for i in range(4):
            key_conflict_i = f"E={i}, dm_conflict_out"
            key_align_i = f"E={i}, dm_align_out"   
            pred_conflict_i = logits_all[key_conflict_i]
            pred_align_i = logits_all[key_align_i]    
            loss_dis_conflict_i_ = self.intri_criterion(pred_conflict_i, label).detach()
            loss_dis_align_i_ = self.intri_criterion(pred_align_i, label).detach()

            # EMA sample loss
            getattr(self, f'sample_loss_ema_d{i}').update(loss_dis_conflict_i_, index)
            getattr(self, f'sample_loss_ema_b{i}').update(loss_dis_align_i_, index)

            # class-wise normalize
            loss_dis_conflict_i_ = getattr(self, f'sample_loss_ema_d{i}').parameter[index].clone().detach()
            loss_dis_align_i_ = getattr(self, f'sample_loss_ema_b{i}').parameter[index].clone().detach()

            loss_dis_conflict_i_ = loss_dis_conflict_i_.cuda()
            loss_dis_align_i_ = loss_dis_align_i_.cuda()

            for c in range(self.num_classes):
                class_index = torch.where(label == c)[0].cuda()
                max_loss_conflict = getattr(self, f'sample_loss_ema_d{i}').max_loss(c)
                max_loss_align = getattr(self, f'sample_loss_ema_b{i}').max_loss(c)
                loss_dis_conflict_i_[class_index] /= max_loss_conflict
                loss_dis_align_i_[class_index] /= max_loss_align

            loss_weight_i  = loss_dis_align_i_ / (loss_dis_align_i_+ loss_dis_conflict_i_ + 1e-8)  
            loss_dis_conflict_i = self.intri_criterion(pred_conflict_i, label) * loss_weight_i.cuda() 
            loss_dis_align_i = self.bias_criterion(pred_align_i, label)
            loss_dis_conflict_all[i,:] = loss_dis_conflict_i
            loss_dis_align_all[i,:] = loss_dis_align_i                              
            key_out_mix = f"E={i}, dm_out_mix" 
            key_indices_mini_i = f"E={i}, indices_mini" 
            indices_mini_i = logits_all[key_indices_mini_i] 
            pred_out_mix = logits_all[key_out_mix] 
            loss_cf_i = self.contrastive_loss(pred_out_mix, pred_conflict_i, indices_mini_i)                                               
            loss_cf[i,:] = loss_cf_i
            
        kl_loss_conflict_1 = self.kl_loss(logits_all['E=1, dm_align_out'], logits_all['E=0, dm_align_out'].detach())
        kl_loss_conflict_2 = self.kl_loss(logits_all['E=2, dm_align_out'], logits_all['E=1, dm_align_out'].detach())         
        kl_loss_conflict_3 = self.kl_loss(logits_all['E=3, dm_align_out'], logits_all['E=2, dm_align_out'].detach())  
        kl_loss_conflict = kl_loss_conflict_1 + kl_loss_conflict_2 + kl_loss_conflict_3
                    
        loss_dis_experts  = self.alpha2*loss_dis_conflict_all.mean() +  loss_dis_align_all.mean()
        loss_cf_experts = loss_cf.mean()
        loss_experts = 4*loss_dis_experts + self.beta * loss_cf_experts + kl_loss_conflict                                                    # Eq.4 Total objective
    
        ###### gate 
        pred_conflict = logits_gate['dm_conflict_out']
        loss_dis_conflict = self.intri_criterion(pred_conflict, label)
        loss_gate  = loss_dis_conflict.mean()
        
        loss = loss_experts + loss_gate              
        loss.backward()
        self.optimizer.step()   
        
        pred = pred_conflict.data.max(1, keepdim=True)[1].squeeze(1)

        step_vals = {'loss':loss.item()}
        if cf_matrix is None:
            return step_vals
        cf_train = cf_matrix(pred.cpu(),label.cpu())
        return step_vals, cf_train


class UBNet(Algorithm):
    """
    https://github.com/aandyjeon/UBNet
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.orthnet = OrthoNet(num_classes=num_classes, arch=hparams['arch'])
        self.loss_orth = nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.orthnet.parameters()), lr=self.hparams["lr"],  weight_decay=self.hparams['weight_decay'])
        if hparams['arch'] == 'resnet18':
            self.network = torchvision.models.resnet18(weights=hparams['pretrain'])
            self.n_outputs = 512
        elif hparams['arch'] == 'resnet50':
            self.network = torchvision.models.resnet50(weights=hparams['pretrain'])
            self.n_outputs = 2048
        nc = input_shape[0]
        
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        self.network.fc = networks.Classifier(
            self.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.criterion = nn.CrossEntropyLoss()

    def init_dataloader(self, train_data, val_data):

        self.loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=train_data.N_WORKERS,
            pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=train_data.N_WORKERS,
            pin_memory=True)
    
    def save_base_model(self):
        torch.save({
                'state_dict': self.network.state_dict(), 
            }, os.path.join(self.output_dir, 'base_model.pkl'))
        
    def train_base_model(self):
        print('Training the base model...')
        self.best_val_acc = 0
        earlystopper = EarlyStopper(patience=self.hparams['base_patience'])
        self.base_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        for epoch in range(self.hparams['base_epochs']):
            
            self.network.train()
            for data, target in self.loader:#in tqdm(self.loader, desc='Training the base model', ncols=35):
                data, target = data.cuda(), target.cuda()
                out = self.network(data)
                loss = self.criterion(out,target)
                
                self.base_optimizer.zero_grad()
                loss.backward()
                self.base_optimizer.step()
            
            self.network.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in self.val_loader:#, desc='Running validation', ncols=35):
                    data, target = data.cuda(), target.cuda()
                    out = self.network(data)
                    B = target.size(0)
                    correct += (out.argmax(1).eq(target).float()).sum().item()
                    total+=B
            val_acc = correct/total

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_base_model()

            print('Base model training... Epoch: {}/{}, Current Val Acc: {:.4f}, Best Acc: {:.4f}'.format(
                epoch,self.hparams['base_epochs'], val_acc, self.best_val_acc))
            
            if earlystopper.early_stop(val_acc) or val_acc == 1.0:
                break
            
        val_checkpoint = torch.load(os.path.join(self.output_dir, 'base_model.pkl'))
        self.network.load_state_dict(val_checkpoint['state_dict'])
        os.remove(os.path.join(self.output_dir, 'base_model.pkl'))

    def eval(self):
        self.network.eval()
        self.orthnet.eval()

    def update_cf(self, minibatches, unlabeled=None, cf_matrix=None):

        self.network.train()
        self.orthnet.train()
        extractor_1 = nn.Sequential(*list(self.network.children())[:4]).cuda()
        extractor_2 = nn.Sequential(*list(self.network.children())[:5]).cuda()
        extractor_3 = nn.Sequential(*list(self.network.children())[:6]).cuda()
        extractor_4 = nn.Sequential(*list(self.network.children())[:7]).cuda()
        extractor_5 = nn.Sequential(*list(self.network.children())[:8]).cuda()
        images = torch.cat([x for x, y in minibatches])
        labels = torch.cat([y for x, y in minibatches])
        
        for param in extractor_1.parameters():
            param.requires_grad = False
        for param in extractor_2.parameters():
            param.requires_grad = False
        for param in extractor_3.parameters():
            param.requires_grad = False
        for param in extractor_4.parameters():
            param.requires_grad = False
        for param in extractor_5.parameters():
            param.requires_grad = False

        feature_1 = extractor_1.forward(images)
        feature_2 = extractor_2.forward(images)
        feature_3 = extractor_3.forward(images)
        feature_4 = extractor_4.forward(images)
        feature_5 = extractor_5.forward(images)
        
        out = {}
        out['out1'] = feature_1
        out['out2'] = feature_2
        out['out3'] = feature_3
        out['out4'] = feature_4
        out['out5'] = feature_5
        self.optimizer.zero_grad()
        #print(images.shape,feature_1.shape,feature_2.shape,feature_3.shape,feature_4.shape,feature_5.shape)
        pred_label_orth, loss_conv, loss_trans = self.orthnet(out)
        loss_orth = self.loss_orth(pred_label_orth, labels)

        loss_orth.backward()
        self.optimizer.step()
        loss_total = loss_orth+loss_conv+loss_trans
        step_vals = {'loss':loss_total.item()}
        if cf_matrix is None:
            return step_vals
        cf_train = cf_matrix(pred_label_orth.argmax(1).cpu(),labels.cpu())
        return step_vals, cf_train
    
    def predict(self, x):
        extractor_1 = nn.Sequential(*list(self.network.children())[:4]).cuda()
        extractor_2 = nn.Sequential(*list(self.network.children())[:5]).cuda()
        extractor_3 = nn.Sequential(*list(self.network.children())[:6]).cuda()
        extractor_4 = nn.Sequential(*list(self.network.children())[:7]).cuda()
        extractor_5 = nn.Sequential(*list(self.network.children())[:8]).cuda()
        
        for param in extractor_1.parameters():
            param.requires_grad = False
        for param in extractor_2.parameters():
            param.requires_grad = False
        for param in extractor_3.parameters():
            param.requires_grad = False
        for param in extractor_4.parameters():
            param.requires_grad = False
        for param in extractor_5.parameters():
            param.requires_grad = False

        feature_1 = extractor_1.forward(x)
        feature_2 = extractor_2.forward(x)
        feature_3 = extractor_3.forward(x)
        feature_4 = extractor_4.forward(x)
        feature_5 = extractor_5.forward(x)
        
        out = {}
        out['out1'] = feature_1
        out['out2'] = feature_2
        out['out3'] = feature_3
        out['out4'] = feature_4
        out['out5'] = feature_5
        
        pred_label_orth, _,_ = self.orthnet(out)

        return pred_label_orth

    def init_centroids(self):
        pass


class OccamNets(Algorithm):
    """
    https://github.com/erobic/occam-nets-v1
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.best_valid_acc = 0
        
        self.model = OccamResNet(input_shape, hparams, exits_kwargs=None, num_classes=num_classes).cuda()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.num_exits = len(self.model.multi_exit.exit_block_nums)
        self.init_losses()

    def init_losses(self):
        gate_cfg = self.hparams['exit_gating']
        for exit_ix in range(4):
            if gate_cfg['loss_wt'] != 0.0:
                loss_name = f'ExitGateLoss_{exit_ix}'
                # The loss is stateful (maintains accuracy/gate_wt)
                if not hasattr(self, loss_name):
                    setattr(self, loss_name, ExitGateLoss(gate_cfg['train_acc_thresholds'][exit_ix],
                                                        gate_cfg['balance_factor']))
                getattr(self, loss_name).on_epoch_start()
            loss_name = f"GateWeightedCELoss_{exit_ix}"
            setattr(self, loss_name, GateWeightedCELoss(gate_cfg['gamma0'], gate_cfg['gamma'], offset=gate_cfg['weight_offset']))

    def predict(self, x):
        return self.model(x)['E=3, logits']
    
    def update_cf(self, minibatches, unlabeled=None, cf_matrix=None):
        all_x = torch.cat([x for x, _, _ in minibatches])
        all_y = torch.cat([y for _, y, _ in minibatches])
        index = torch.cat([z for _, _, z in minibatches])
        model_out = self.model(all_x)
        loss = 0

        # Compute exit-wise losses
        for exit_ix in range(len(self.model.multi_exit.exit_block_nums)):
            _loss_dict = self.compute_losses(all_y, model_out, exit_ix, index)
            for _k in _loss_dict:
                #self.log(f'{_k} E={exit_ix}', _loss_dict[_k].mean(), py_logging=False)
                loss += _loss_dict[_k].mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        p = model_out['E=3, logits']
        if cf_matrix is not None:
            cf_train = cf_matrix(p.argmax(1).cpu(),all_y.cpu())
    
        step_vals = {'loss': loss.item()}
        if cf_matrix is None:
            return step_vals
        return step_vals, cf_train
    
    def compute_losses(self, y, model_out, exit_ix, index):
        """
        Computes CAM Suppression loss, exit gate loss and gate-weighted CE Loss
        :param batch:
        :param batch_idx:
        :param model_out:
        :param exit_ix:
        :return:
        """
        gt_ys = y.squeeze()
        loss_dict = {}

        logits = model_out[f'E={exit_ix}, logits']

        ###############################################################################################################
        # Compute CAM suppression loss
        ###############################################################################################################
        supp_cfg = self.hparams['cam_suppression']
        if supp_cfg['loss_wt'] != 0.0:
            loss_dict['supp'] = supp_cfg['loss_wt'] * CAMSuppressionLoss()(model_out[f'E={exit_ix}, cam'], gt_ys)

        ###############################################################################################################
        # Compute exit gate loss
        ###############################################################################################################
        gate_cfg = self.hparams['exit_gating']
        if gate_cfg['loss_wt'] != 0.0:
            loss_name = f'ExitGateLoss_{exit_ix}'
            gates = model_out[f'E={exit_ix}, gates']
            force_use = (self.current_epoch + 1) <= gate_cfg['min_epochs']
            loss_dict['gate'] = gate_cfg['loss_wt'] * getattr(self, loss_name) \
                (index, logits, gt_ys, gates, force_use=force_use)

        ###############################################################################################################
        # Compute gate-weighted CE Loss
        ###############################################################################################################

        loss_name = f"GateWeightedCELoss_{exit_ix}"
        prev_gates = None if exit_ix == 0 else model_out[f"E={exit_ix - 1}, gates"]
        unweighted_loss = self.compute_main_loss(y, model_out, exit_ix)
        assert len(unweighted_loss) == len(y)
        loss_dict['main'] = getattr(self, loss_name)(exit_ix, logits, prev_gates, gt_ys, unweighted_loss)
        return loss_dict

    def compute_main_loss(self, y, model_out, exit_ix):
        return F.cross_entropy(model_out[f'E={exit_ix}, logits'], y.squeeze(), reduction='none')