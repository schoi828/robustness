# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np
from typing import Union, List, Dict, Any, cast

from domainbed.lib import wide_resnet
import copy
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
#from torchvision.ops. import Conv2dNormActivation
import math
from torchvision.models.vision_transformer import ConvStemConfig, Conv2dNormActivation, Encoder,_log_api_usage_once
from torch.utils import model_zoo
from domainbed.lib.misc import (
    reparametrize
)
from domainbed.lib.modules import *
from domainbed.lib.occam_lib import Occam_MultiExitModule
from domainbed.lib.variable_width_resnet import VariableWidthResNet, BasicBlock, Bottleneck

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()

        if hparams['arch'] == 'resnet18':
            self.network = torchvision.models.resnet18(weights=hparams['pretrain'])
            self.n_outputs = 512
        elif hparams['arch'] == 'resnet50':
            self.network = torchvision.models.resnet50(weights=hparams['pretrain'])
            self.n_outputs = 2048
        elif hparams['arch'] == 'resnet101':
            self.network = torchvision.models.resnet101(weights=hparams['pretrain'])
            self.n_outputs = 2048
        else:
            raise NotImplementedError

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class B_VAE(nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        ## define VAE
        self.latent_dim = 256
        self.beta = 0.5
        self.gamma = 500
        self.loss_type = 'H'
        self.C_max = torch.Tensor([25])
        self.C_stop_iter = 1e5
        self.kld_weight = self.latent_dim / np.prod(input_shape)  # Account for the minibatch samples from the dataset
        modules = []
        hidden_dims = [16, 32, 64, 128, 256, 512, 512]
        strides = [1, 2, 1, 2, 1, 2, 1]
        ksizes = [5, 4, 3, 3, 4, 4, 4]
        if input_shape[1] == 96:
            hidden_dims.append(512)
            strides.append(2)
            ksizes.append(5)
        if input_shape[1] == 256:
            hidden_dims+=[512,512,512]
            strides+=[2,2,2]
            ksizes+=[5,5,4]
        if input_shape[1] == 128:
            hidden_dims+=[512,512]
            strides+=[2,2]
            ksizes+=[5,3]
        # Build Encoder
        in_channels = input_shape[0]
        for h_dim, stride, ksize in zip(hidden_dims, strides, ksizes):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=ksize, stride=stride, padding=0),
                    # nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], self.latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1])

        hidden_dims.reverse()
        strides.reverse()
        ksizes.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=ksizes[i],
                                       stride=strides[i],
                                       padding=0,
                                       output_padding=0),
                    # nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               input_shape[0],
                                               kernel_size=ksizes[-1],
                                               stride=strides[-1],
                                               padding=0,
                                               output_padding=0),
                            # nn.BatchNorm2d(hidden_dims[-1]),
                            # nn.LeakyReLU(),
                            # nn.Conv2d(hidden_dims[-1], out_channels= 3,
                            #           kernel_size= 3, padding= 1),
                            nn.Tanh()
                        )
        
    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def forward(self, x):
        return self.forward_vae(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward_latent(self, input: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(input)
        return self.reparameterize(mu, log_var)

    def forward_vae(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def update_vae(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        recons, x, mu, log_var = self.forward_vae(x)
        recons_loss = F.mse_loss(recons, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            vae_loss = recons_loss + self.beta * self.kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(x.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            vae_loss = recons_loss + self.gamma * self.kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'vae_loss': vae_loss, 'Reconstruction_Loss':recons_loss.item(), 'KLD':kld_loss.item()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward_vae(x)[0]
    

class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    
    # for resnet18, resnet50, resnet101
    if 'resnet' in hparams['arch']: 
        return ResNet(input_shape, hparams)
    elif 'vit' in hparams['arch']:
        raise NotImplementedError
    else:
        raise NotImplementedError

def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929.
    Version from torchvision.model
    Unlike the original torchvision version, it can process grayscale images

    """

    def __init__(
        self,
        image_size: int,
        in_ch:int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=in_ch, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x
    

class AugNet(nn.Module):
    def __init__(self, noise_lv, img_shape):
        super(AugNet, self).__init__()
        ############# Trainable Parameters
        img_shape = (3,img_shape[1],img_shape[2])
        self.noise_lv = nn.Parameter(torch.zeros(1))
        k_list = [9,13,17,5]
        for i, k in enumerate(k_list):
                     
            var_name = 'shift_var'            
            mean_name = 'shift_mean'
            if i > 0:
                var_name+=str(i+1)
                mean_name+=str(i+1)
            shape = (3,img_shape[1]-k+1,img_shape[2]-k+1)

            setattr(self,var_name, nn.Parameter(torch.empty(shape)))
            nn.init.normal_(getattr(self,var_name), 1, 0.1)
            setattr(self,mean_name, nn.Parameter(torch.zeros(shape)))
            nn.init.normal_(getattr(self,mean_name), 0, 0.1)
        

        self.norm = nn.InstanceNorm2d(3)

        ############## Fixed Parameters (For MI estimation
        self.spatial = nn.Conv2d(3, 3, 9).cuda()
        self.spatial_up = nn.ConvTranspose2d(3, 3, 9).cuda()

        self.spatial2 = nn.Conv2d(3, 3, 13).cuda()
        self.spatial_up2 = nn.ConvTranspose2d(3, 3, 13).cuda()

        self.spatial3 = nn.Conv2d(3, 3, 17).cuda()
        self.spatial_up3 = nn.ConvTranspose2d(3, 3, 17).cuda()


        self.spatial4 = nn.Conv2d(3, 3, 5).cuda()
        self.spatial_up4 = nn.ConvTranspose2d(3, 3, 5).cuda()

        self.color = nn.Conv2d(3, 3, 1).cuda()

        for param in list(list(self.color.parameters()) +
                          list(self.spatial.parameters()) + list(self.spatial_up.parameters()) +
                          list(self.spatial2.parameters()) + list(self.spatial_up2.parameters()) +
                          list(self.spatial3.parameters()) + list(self.spatial_up3.parameters()) +
                          list(self.spatial4.parameters()) + list(self.spatial_up4.parameters())
                          ):
            param.requires_grad=False

    def forward(self, x, estimation=False):

        if not estimation:
            spatial = nn.Conv2d(3, 3, 9).cuda()
            spatial_up = nn.ConvTranspose2d(3, 3, 9).cuda()

            spatial2 = nn.Conv2d(3, 3, 13).cuda()
            spatial_up2 = nn.ConvTranspose2d(3, 3, 13).cuda()

            spatial3 = nn.Conv2d(3, 3, 17).cuda()
            spatial_up3 = nn.ConvTranspose2d(3, 3, 17).cuda()

            spatial4 = nn.Conv2d(3, 3, 5).cuda()
            spatial_up4 = nn.ConvTranspose2d(3, 3, 5).cuda()

            color = nn.Conv2d(3,3,1).cuda()
            weight = torch.randn(5)

            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(F.dropout(color(x), p=.2))

            x_sdown = spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(spatial_up(x_sdown))
            #
            x_s2down = spatial2(x)

            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(spatial_up2(x_s2down))
            #
            #
            x_s3down = spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(spatial_up3(x_s3down))

            #
            x_s4down = spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.tanh(spatial_up4(x_s4down))

            output = (weight[0] * x_c + weight[1] * x_s + weight[2] * x_s2+ weight[3] * x_s3 + weight[4]*x_s4) / weight.sum()
        else:
            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(self.color(x))
            #
            x_sdown = self.spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(self.spatial_up(x_sdown))
            #
            x_s2down = self.spatial2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(self.spatial_up2(x_s2down))

            x_s3down = self.spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(self.spatial_up3(x_s3down))

            x_s4down = self.spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.tanh(self.spatial_up4(x_s4down))

            output = (x_c + x_s + x_s2 + x_s3 + x_s4) / 5
        return output
    

class L2D_ResNet(nn.Module):
    def __init__(self, featurizer, num_classes):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = nn.Linear(512,num_classes)#classifier

        self.p_logvar = nn.Sequential(nn.Linear(self.featurizer.n_outputs, 512),
                                      nn.ReLU())
        self.p_mu = nn.Sequential(nn.Linear(self.featurizer.n_outputs, 512),
                                  nn.LeakyReLU())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x, gt=None, train=True, classifiy=False, **kwargs):
        end_points = {}
        x = self.featurizer(x).clamp(-100,100)

        logvar = torch.nan_to_num(self.p_logvar(x))
        mu = torch.nan_to_num(self.p_mu(x))
        #
        end_points['logvar'] = logvar
        end_points['mu'] = mu

        if train:
            x = reparametrize(mu, logvar)
        else:
            x = mu
        x = torch.nan_to_num(x)
        end_points['Embedding'] = x
        x = self.classifier(x)
        end_points['Predictions'] = nn.functional.softmax(input=x, dim=-1)

        return x, end_points


class Centroids(nn.Module):
    
    def __init__(self, hparams, num_classes, per_clusters, feature_dim=None):
        super(Centroids, self).__init__()
        self.hparams = hparams
        self.momentum = hparams['momentum']
        self.per_clusters = per_clusters
        self.num_classes = num_classes
        self.feature_dim = hparams['feature_dim'] if feature_dim is None else feature_dim
        
        # Cluster
        self.cluster_means = None
        self.cluster_vars = torch.zeros((self.num_classes, self.per_clusters))
        self.cluster_losses = torch.zeros((self.num_classes, self.per_clusters))
        self.cluster_accs = torch.zeros((self.num_classes, self.per_clusters))
        self.cluster_weights = torch.zeros((self.num_classes, self.per_clusters))
        
        # Sample
        self.feature_bank = None
        self.assigns = None
        self.corrects = None
        self.losses = None
        self.weights = None
        
        self.initialized = False
        self.weight_type = 'scale_loss'#args.cluster_weight_type
        
        self.max_cluster_weights = hparams['max_weight']#100. # 0 means no-limit

    def __repr__(self):
        return "{}(Y{}/K{}/dim{})".format(self.__class__.__name__, self.num_classes, self.per_clusters, self.feature_dim)
    
    @property
    def num_clusters(self):
        return self.num_classes * self.per_clusters
            
    @property 
    def cluster_counts(self):
        if self.assigns is None:
            return 0
        return self.assigns.bincount(minlength=self.num_clusters)
    
    
    def _clamp_weights(self, weights):
        if self.max_cluster_weights > 0:
            if weights.max() > self.max_cluster_weights:
                scale = np.log(self.max_cluster_weights)/torch.log(weights.cpu().max())
                scale = scale.cuda()
                #print("> Weight : {:.4f}, scale : {:.4f}".format(weights.max(), scale))
                return weights ** scale
        return weights
        
    
    def get_cluster_weights(self, ids):
        if self.assigns is None:
            return 1
        
        cluster_counts = self.cluster_counts + (self.cluster_counts==0).float() # avoid nans
            
        cluster_weights = cluster_counts.sum()/(cluster_counts.float())
        assigns_id = self.assigns[ids]

        if (self.losses == -1).nonzero().size(0) == 0:
            cluster_losses_ = self.cluster_losses.view(-1)
            losses_weight = cluster_losses_.float()/cluster_losses_.sum()
            weights_ = cluster_weights[assigns_id] * losses_weight[assigns_id.cpu()].cuda()
            weights_ /= weights_.mean()
        else:
            weights_ = cluster_weights[assigns_id]
            weights_ /= weights_.mean()
        
        return self._clamp_weights(weights_)
    
    
    def initialize_(self, cluster_assigns, cluster_centers, sorted_features=None):
        cluster_means = cluster_centers.detach().cuda()
        cluster_means = F.normalize(cluster_means, 1)
        self.cluster_means = cluster_means.view(self.num_classes, self.per_clusters, -1)
        
        N = cluster_assigns.size(0)
        self.feature_bank = torch.zeros((N, self.feature_dim)).cuda() if sorted_features is None else sorted_features
        self.assigns = cluster_assigns
        self.corrects = torch.zeros((N)).long().cuda() - 1
        self.losses = torch.zeros((N)).cuda() - 1
        self.weights = torch.ones((N)).cuda()
        self.initialized = True
        
        
    def get_variances(self, x, y):
        return 1 - (y @ x).mean(0)
        
    def compute_centroids(self, verbose=False, split=False):
        for y in range(self.num_classes):
            for k in range(self.per_clusters): 
                l = y*self.per_clusters + k
                ids = (self.assigns==l).nonzero()
                if ids.size(0) == 0:
                    continue
                self.cluster_means[y, k] = self.feature_bank[ids].mean(0)
                self.cluster_vars[y, k] = self.get_variances(self.cluster_means[y, k], self.feature_bank[ids])

                corrs = self.corrects[ids]
                corrs_nz = (corrs[:, 0]>=0).nonzero()
                if corrs_nz.size(0) > 0:
                    self.cluster_accs[y, k] = corrs[corrs_nz].float().mean(0)

                losses = self.losses[ids]
                loss_nz = (losses[:, 0]>=0).nonzero()
                if loss_nz.size(0) > 0:
                    self.cluster_losses[y, k] = losses[loss_nz].float().mean(0)
            
        return 

                 
    def update(self, results, target, ids, features=None):
        assert self.initialized
        
        ### update feature and assigns
        feature = results["feature"] if features is None else features
        feature_ = F.normalize(feature, 1).detach()
       
        feature_new = (1-self.momentum) * self.feature_bank[ids] + self.momentum * feature_
        feature_new = F.normalize(feature_new, 1)
        
        self.feature_bank[ids] = feature_new

        sim_score = self.cluster_means @ feature_new.permute(1, 0) # YKC/CB => YKB

        for y in range(self.num_classes):
            sim_score[y, :, (target!=y).nonzero()] -= 1e4
            
        sim_score_ = sim_score.view(self.num_clusters, -1)
        new_assigns = sim_score_.argmax(0)
        self.assigns[ids] = new_assigns
        
        corrects = (results["out"].argmax(1) == target).long()
        self.corrects[ids] = corrects
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        losses = criterion(results["out"], target.long()).detach()
        self.losses[ids] = losses
        
        return
    

        
class FixedCentroids(Centroids):
        
    def compute_centroids(self, verbose='', split=False):
        
        for y in range(self.num_classes):
            for k in range(self.per_clusters): 
                l = y*self.per_clusters + k
                
                ids = (self.assigns==l).nonzero()
                if ids.size(0) == 0:
                    continue

                corrs = self.corrects[ids]
                corrs_nz = (corrs[:, 0]>=0).nonzero()
                if corrs_nz.size(0) > 0:
                    self.cluster_accs[y, k] = corrs[corrs_nz].float().mean(0)

                losses = self.losses[ids]
                loss_nz = (losses[:, 0]>=0).nonzero()
                if loss_nz.size(0) > 0:
                    self.cluster_losses[y, k] = losses[loss_nz].float().mean(0)
                    
                self.cluster_weights[y, k] = self.weights[ids].float().mean(0)
     
        return 
                
    def get_cluster_weights(self, ids):
        weights_ids = super().get_cluster_weights(ids)
        self.weights[ids] = weights_ids
        return weights_ids

    
    def update(self, results, target, ids, features=None, preds=None):
        assert self.initialized
        
        out = preds if preds is not None else results["out"]
        
        corrects = (out.argmax(1) == target).long()
        self.corrects[ids] = corrects
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        losses = criterion(out, target.long()).detach()
        self.losses[ids] = losses
        
        return
    
    
    
class AvgFixedCentroids(FixedCentroids):
    
    def __init__(self, hparams, num_classes, per_clusters, feature_dim=None):
        super(AvgFixedCentroids, self).__init__(hparams, num_classes, per_clusters, feature_dim)
        self.exp_step = hparams['exp_step']
        self.avg_weight_type = 'expavg'#hparams['avg_weight_type']
        
    def compute_centroids(self, verbose='', split=False):
        
        for y in range(self.num_classes):
            for k in range(self.per_clusters): 
                l = y*self.per_clusters + k
                
                ids = (self.assigns==l).nonzero()
                if ids.size(0) == 0:
                    continue

                corrs = self.corrects[ids]
                corrs_nz = (corrs[:, 0]>=0).nonzero()
                if corrs_nz.size(0) > 0:
                    self.cluster_accs[y, k] = corrs[corrs_nz].float().mean(0)

                losses = self.losses[ids]
                loss_nz = (losses[:, 0]>=0).nonzero()
                if loss_nz.size(0) > 0:
                    self.cluster_losses[y, k] = losses[loss_nz].float().mean(0)
                    
                self.cluster_weights[y, k] = self.weights[ids].float().mean(0)
                    
        return 
        
        
    def get_cluster_weights(self, ids):
        
        weights_ids = super().get_cluster_weights(ids)
        
        if self.avg_weight_type == 'expavg':
            weights_ids_ = self.weights[ids] * torch.exp(self.exp_step*weights_ids.data)
        elif self.avg_weight_type == 'avg':
            weights_ids_ = (1-self.momentum) * self.weights[ids] + self.momentum * weights_ids
        elif self.avg_weight_type == 'expgrad':
            weights_ids_l1 = weights_ids / weights_ids.sum()
            prev_ids_l1 = self.weights[ids] / self.weights[ids].sum()
            weights_ids_ = prev_ids_l1 * torch.exp(self.exp_step*weights_ids_l1.data)
        else:
            raise ValueError
            
        self.weights[ids] = weights_ids_ / weights_ids_.mean()
        return self.weights[ids]  




class PnDNet(nn.Module):
    def __init__(self, input_shape, num_classes=10, pretrained=True, multi_exit_type=MultiExitModule,exits_kwargs={},hparams=None):
        super(PnDNet, self).__init__()
        self.num_classes = num_classes
        
        if hparams['arch'] == 'resnet18':
            model_d = torchvision.models.resnet18(weights=hparams['pretrain'])
            self.n_outputs = 512
        elif hparams['arch'] == 'resnet50':
            model_d = torchvision.models.resnet50(weights=hparams['pretrain'])
            self.n_outputs = 2048
        else:
            raise Exception
        
        self.arch = hparams['arch']
        nc = input_shape[0]
        if nc != 3:
            tmp = model_d.conv1.weight.data.clone()

            model_d.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                model_d.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]
        model_b = copy.deepcopy(model_d)

        self.model_d = nn.Sequential(*list(model_d.children()))[:-1]
        self.model_b = nn.Sequential(*list(model_b.children()))[:-1]
        
        exits_kwargs['exit_out_dims'] = num_classes
        if self.arch == 'resnet18':
            out_dims = [64, 128, 256, 512]
        else:
            out_dims = [256, 512, 1024, 2048]
        self.exits_cfg = exits_kwargs
        multi_exit = multi_exit_type(**exits_kwargs)
        for i in range(0, 4):
            multi_exit.build_and_add_exit(out_dims[i])
        self.multi_exit = multi_exit

        self.gate_fc = nn.Linear(self.exits_cfg['exit_out_dims']*4, 4)

    def forward(self, x, y = None, use_mix = False, feature=False):
        block_num_to_exit_in_i = {}
        block_num_to_exit_in_b = {}
        x_d = self.model_d[:4](x)
        for i in range(0, 4):
            x_d = self.model_d[4+i](x_d)
            block_num_to_exit_in_i[i] = x_d     

        ##### bias network
        x_b = self.model_b[:4](x)
        for i in range(0, 4):
            x_b = self.model_b[4+i](x_b)
            block_num_to_exit_in_b[i] = x_b     

        if feature:
            return block_num_to_exit_in_i
        ##### mutiple modules  
        each_block_out = self.multi_exit(block_num_to_exit_in_i, block_num_to_exit_in_b, y, use_mix = use_mix)

        final_out = {'dm_conflict_out': 0 }
        out_logit_names = ['dm_conflict_out']      
        for out_logit_name in out_logit_names:
            exit_0 = f"E=0, {out_logit_name}"
            exit_1 = f"E=1, {out_logit_name}"
            exit_2 = f"E=2, {out_logit_name}"
            exit_3 = f"E=3, {out_logit_name}"
            
            ##### gating network  
            gate_in = torch.cat((each_block_out[exit_0],each_block_out[exit_1],each_block_out[exit_2],each_block_out[exit_3]),1)
            x_gate = self.gate_fc(gate_in)
            pr_gate = F.softmax(x_gate, dim=1)         
            logits_gate_i = torch.stack([each_block_out[exit_0].detach(), each_block_out[exit_1].detach(), each_block_out[exit_2].detach(), each_block_out[exit_3].detach()], dim=-1)
            logits_gate_i = logits_gate_i * pr_gate.view(pr_gate.size(0), 1, pr_gate.size(1))
            logits_gate_i = logits_gate_i.sum(-1)
            final_out[out_logit_name] = logits_gate_i            
             
        return each_block_out,final_out
    

class OrthoNet(nn.Module):
    def __init__(self, num_classes: int = 2, arch='resnet18') :
        super(OrthoNet, self).__init__()
        self.num_classes = num_classes

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.test = nn.Conv2d(64,64,kernel_size = 1)

        if arch=='resnet18':
            trans_dims = [64,64,128,256,512]
        else:
            trans_dims = [64,256,512,1024,2048]
        self.trans1 = nn.Sequential(
            nn.Conv2d(trans_dims[0], 64, kernel_size = 1),
            nn.LeakyReLU(0.1)
        )
        self.trans2 = nn.Sequential(
            nn.Conv2d(trans_dims[1], 64, kernel_size = 1),
            nn.LeakyReLU(0.1)
        )
        self.trans3 = nn.Sequential(
            nn.Conv2d(trans_dims[2], 64, kernel_size = 1),
            nn.LeakyReLU(0.1)
        )
        self.trans4 = nn.Sequential(
            nn.Conv2d(trans_dims[3], 64, kernel_size = 1),
            nn.LeakyReLU(0.1)
        )
        self.trans5 = nn.Sequential(
            nn.Conv2d(trans_dims[4], 64, kernel_size = 1),
            nn.LeakyReLU(0.1)
        )

        self.reduction = CodeReduction(c_dim = 64*5,  feat_hw = 7, blocks = 5) 

        self.fc1 = nn.Linear(64, self.num_classes)
        self.fc2 = nn.Linear(64, self.num_classes)
        self.fc3 = nn.Linear(64, self.num_classes)
        self.fc4 = nn.Linear(64, self.num_classes)
        self.fc5 = nn.Linear(64, self.num_classes)

        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x: Dict[str, torch.Tensor])->torch.Tensor:

        x1, x2, x3, x4, x5 = x['out1'], x['out2'], x['out3'], x['out4'], x['out5']

        out1 = self.avgpool(x1) 
        out1 = self.trans1(out1) 
        
        out2 = self.avgpool(x2)
        out2 = self.trans2(out2)

        out3 = self.avgpool(x3)
        out3 = self.trans3(out3)

        out4 = self.avgpool(x4)
        out4 = self.trans4(out4)

        out5 = self.avgpool(x5)
        out5 = self.trans5(out5)
        
        out_concat = torch.cat((out1, out2, out3, out4, out5), axis = 1) 
        out, loss_conv, loss_trans = self.reduction(out_concat)

        out1_, out2_, out3_, out4_, out5_= torch.split(out, [out.shape[1]//5]*5, dim = 1)

        out1_ = self.fc1(out1_)
        out2_ = self.fc2(out2_)
        out3_ = self.fc3(out3_)
        out4_ = self.fc4(out4_)
        out5_ = self.fc5(out5_)

        out = (out1_ + out2_ + out3_ + out4_ + out5_)/5
        out = self.softmax(out)


        return out, loss_conv, loss_trans
    

class OccamResNet(ResNet):
    def __init__(self, input_shape, hparams, num_classes, exits_kwargs=None) -> None:
        """
        Adds multiple exits to DenseNet
        :param width:
        :param exits_kwargs: all the parameters needed to create the exits

        """
        super().__init__(input_shape, hparams)
        self.exits_cfg = exits_kwargs
        del self.network.fc
        base_dim_dict = {'resnet18':64, 'resnet50':256}
        multi_exit = Occam_MultiExitModule(exit_out_dims=num_classes)
        for i in range(0, 4):
            multi_exit.build_and_add_exit(base_dim_dict[hparams['arch']]*(2**i))
        self.multi_exit = multi_exit

    def forward(self, x, y=None):
        block_num_to_exit_in = {}
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        if hasattr(self.network, "maxpool"):
            x = self.network.maxpool(x)

        for i in range(0, 4):
            x = getattr(self.network, f'layer{i + 1}')(x)
            block_num_to_exit_in[i] = x

        return self.multi_exit(block_num_to_exit_in, y=y)

    def get_multi_exit(self):
        return self.multi_exit