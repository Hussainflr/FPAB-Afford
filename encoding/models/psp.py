
from __future__ import division

import torch
import torch.nn as nn

from torch.nn.functional import upsample

from .base import BaseNet
from .fcn import FCNHead
from ..nn import PyramidPooling

class PSP(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PSP, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = PSPHead(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = upsample(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = upsample(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(PyramidPooling(in_channels, norm_layer, up_kwargs),
                                   nn.Conv2d(in_channels * 2, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

def get_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = PSP(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('psp_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_psp_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_psp_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_psp('ade20k', 'resnet50', pretrained, root=root, **kwargs)
