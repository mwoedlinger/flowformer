import torch

import torch.nn as nn
from .modules import PMA, ISAB, SAB, PISAB
from ..base import BaseModel


class LinModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        dim_output = 1
        dim_input = kwargs['_num_markers']

        self.model = nn.Sequential(
            nn.Linear(dim_input, dim_input),
            nn.ReLU()
        )
        self.head = nn.Linear(dim_input, dim_output)

    def forward(self, x):
        y = self.model(x)

        return self.head(y)[:,:,0]

class SetTransformerEncoder(BaseModel):
    """
    Set transformer as described in https://arxiv.org/abs/1810.00825
    dim_input:  dimensionality of input              (flowdata: number of markers)
    num_ouputs: output sequence length               (flowdata: sequence length)
    num_inds:   number of induced points
    dim_hidden: dimension of hidden representation
    num_heads:  number of attention heads
    ln:         use layer norm true/false
    """
    def __init__(self, **kwargs):
        super(SetTransformerEncoder, self).__init__()

        dim_input = kwargs['_num_markers']
        if 'cluster' in kwargs:
            if kwargs['cluster']:
                dim_input = dim_input + dim_input**2 + 1
        self.sequence_length = sequence_length = kwargs['_sequence_length']
        num_inds = kwargs['num_inds']
        dim_hidden = kwargs['dim_hidden'] # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
        num_heads = kwargs['num_heads']
        ln = kwargs['layer_norm']

        enc_layers = [ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln)]
        for _ in range(1, kwargs['hidden_layers']):
            enc_layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        enc_layers.append(ISAB(dim_hidden, dim_input, 1, num_inds, ln=ln)) #num_heads == 1 because dim_input can be a prime number
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        return self.enc(x)

class SetTransformer(BaseModel):
    """
    Set transformer as described in https://arxiv.org/abs/1810.00825
    dim_input:  dimensionality of input              (flowdata: number of markers)
    dim_output: dimensionality of output             (flowdata: 1)
    dim_hidden: dimension of hidden representation
    ln:         use layer norm true/false
    """
    def __init__(self, **kwargs):
        super(SetTransformer, self).__init__()

        dim_input = kwargs['_num_markers']
        if 'cluster' in kwargs:
            if kwargs['cluster']:
                dim_input = dim_input + dim_input**2 + 1
        dim_hidden = kwargs['dim_hidden'] # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
        self.mode = kwargs['mode']
        self.residual = kwargs['residual']
        
        if self.mode == 'autoencoder':
            dim_output = dim_input
        elif self.mode == 'binary':
            dim_output = 1
        else:
            raise NotImplementedError

        assert not (self.mode == 'autoencoder' and self.residual), f'mode is set to {self.mode} while residual connections are used!'

        self.enc = SetTransformerEncoder(**kwargs)
        if self.residual:
            dec_layers = [nn.Linear(2*dim_input, dim_output)]
        else:
            dec_layers = [nn.Linear(dim_input, dim_output)]
        self.dec = nn.Sequential(*dec_layers)


    def forward(self, x):
        enc_out = self.enc(x)
        
        if self.residual:
            dec_in = torch.cat([enc_out, x], dim=-1)
        else:
            dec_in = enc_out

        if self.mode == 'autoencoder':
            return self.dec(dec_in)
        elif self.mode == 'binary':
            return self.dec(dec_in)[:,:,0]


class DebugModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()

        dim_input = kwargs['_num_markers']
        if 'cluster' in kwargs:
            if kwargs['cluster']:
                dim_input = dim_input + dim_input**2 + 1
        conv1d_decoder = kwargs['conv1d_decoder']
        conv1d_kernel = kwargs['conv1d_kernel']
        out_channels = 2

        if len(conv1d_decoder) > 0:
            layers = []
            layers.append(nn.Conv1d(in_channels=dim_input, out_channels=conv1d_decoder[0],
                                    kernel_size=conv1d_kernel[0]))
            layers.append(nn.ReLU())
            for n in range(1, len(conv1d_decoder)):
                layers += [nn.Conv1d(in_channels=conv1d_decoder[n-1], out_channels=conv1d_decoder[n],
                                     kernel_size=conv1d_kernel[n]),
                           nn.ReLU()]
            layers.append(nn.Conv1d(in_channels=conv1d_decoder[-1], out_channels=out_channels, kernel_size=conv1d_kernel[-1]))
            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Conv1d(in_channels=dim_input, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        out = self.model(x.permute((0, 2, 1)))
        return out[:, 0, :]

