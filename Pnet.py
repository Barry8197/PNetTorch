import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import warnings
import captum
from scipy.stats import zscore
import sys
sys.path.insert(0 , '.')

class MaskedLinear(nn.Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        mask : torch.Tensor,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mask = nn.Parameter(
            torch.Tensor(mask.T) , requires_grad = False
        )
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.mask_weights_init()
        #self.weight = nn.Parameter(self.to_sparse(self.weight))
        #self.mask   = nn.Parameter(self.to_sparse(self.mask))

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def mask_weights_init(self) : 
        self.weight.data = self.weight * self.mask
        # Renormalise inital weights to follow kaiming uniform distribution
        non_zero_sum = self.mask.sum()
        if non_zero_sum != 0:
            scaling_factor = self.weight.data.sum() / non_zero_sum
            self.weight.data = self.mask * scaling_factor
        else:
            self.weight.data = self.mask 
            
    def to_sparse(self, tensor):
        """ Convert a dense tensor to a sparse tensor. """
        tensor = tensor.detach()  # Remove from computation graph if needed
        non_zero_indices = tensor.nonzero(as_tuple=False)
        non_zero_values = tensor[non_zero_indices[:, 0], non_zero_indices[:, 1]]
        sparse_tensor = torch.sparse_coo_tensor(non_zero_indices.t(), non_zero_values, tensor.size())
        return sparse_tensor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight * self.mask, self.bias)
        #return torch.sparse.mm(input , (self.weight*self.mask).T) + self.bias

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class PNET(nn.Module):
    def __init__(self, reactome_network, input_dim =None ,  output_dim=None, fcnn=False , activation = nn.ReLU , dropout=0.1 , filter_pathways=False, input_layer_mask = None):
        super().__init__()
        self.reactome_network = reactome_network
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        
        gene_masks, pathway_masks, self.layer_info = self.reactome_network.get_masks(filter_pathways)
                
        if fcnn:
            gene_masks = [np.ones_like(gm) for gm in gene_masks]
            pathway_masks = [np.ones_like(gm) for gm in pathway_masks]
        
        # Prepare list of layers and list of predictions per layer:
        self.layers = nn.ModuleList()
        self.skip = nn.ModuleList()
        
        if input_layer_mask is None : 
            self.layers.append(nn.Linear(in_features = self.input_dim , out_features = gene_masks.shape[0]))
        else : 
            self.layers.append(MaskedLinear(input_layer_mask , in_features=self.input_dim, out_features=gene_masks.shape[0]))
        
        for i in range(0, len(pathway_masks)+1):
            if i ==0 : 
                # Add gene layer first:
                self.layers.append(MaskedLinear(gene_masks , in_features=gene_masks.shape[0],out_features=pathway_masks[i].shape[0]))
                self.skip.append(nn.Linear(in_features=gene_masks.shape[0],out_features=self.output_dim))
            else :
                # Add pathway layers:
                self.layers.append(MaskedLinear(pathway_masks[i-1] , in_features=pathway_masks[i-1].shape[0],out_features=pathway_masks[i-1].shape[1]))
                self.skip.append(nn.Linear(in_features=pathway_masks[i-1].shape[0],out_features=self.output_dim))
                
        # Add final prediction layer:
        self.skip.append(nn.Linear(in_features=pathway_masks[-1].shape[1], out_features=self.output_dim))

    def forward(self, x):
        y = 0
        
        # Iterate through all other pathway layers
        for layer, skip in zip(self.layers, self.skip):
            x =  layer(x)
            y += skip(x)
            
        # Average over output
        y = y/(len(self.layers))
        
        return y

    def deepLIFT_feature_importance(self, test_dataset, target_class=0):
        self.interpret_flag=True
        dl = captum.attr.DeepLift(self)
        feature_importances = dl.attribute((test_dataset), target=target_class)
        if hasattr(self, 'data_index') : 
            data_index = self.data_index
        else : 
            data_index = np.arange(test_dataset.shape[0])
        feature_importances = pd.DataFrame(feature_importances.detach().cpu().numpy(),
                                        index=data_index,
                                        columns=self.features)
        
        self.feature_importances = feature_importances
        self.interpret_flag=False
        return self.feature_importances
    
    def integrated_gradients_feature_importance(self, test_dataset, target_class=0 , task='Classification'):
        self.interpret_flag=True
        ig = captum.attr.IntegratedGradients(self)
        
        if task == 'Regression':
            ig_attr = ig.attribute(test_dataset, n_steps=50)
        else:
            ig_attr, delta = ig.attribute(test_dataset, return_convergence_delta=True, target=target_class)
            
        feature_importances = ig_attr
        
        if hasattr(self, 'data_index') : 
            data_index = self.data_index
        else : 
            data_index = np.arange(test_dataset.shape[0])
            
        feature_importances = pd.DataFrame(feature_importances.detach().cpu().numpy(),
                                            index=data_index,
                                            columns=self.features)
        
        self.feature_importances = feature_importances
        self.interpret_flag=False
        
        return self.feature_importances

    def layerwise_importance(self, test_dataset, target_class=0):
        self.interpret_flag=True
        layer_importance_scores = []
        
        for i, level in enumerate(self.layers):
            print(level)
            cond = captum.attr.LayerConductance(self, level)  # ReLU output of masked layer at each level
            cond_vals = cond.attribute(test_dataset, target=target_class)
            cols = self.layer_info[i]
            if hasattr(self, 'data_index') : 
                data_index = self.data_index
            else : 
                data_index = np.arange(test_dataset.shape[0])
                
            cond_vals_genomic = pd.DataFrame(cond_vals.detach().cpu().numpy(),
                                             columns=cols,
                                             index=data_index)
            pathway_imp_by_target = cond_vals_genomic
            layer_importance_scores.append(pathway_imp_by_target)
        self.interpret_flag=False
        return layer_importance_scores
    
    def layerwise_activation(self, test_dataset, target_class=0):
        self.interpret_flag=True
        layer_importance_scores = []
        
        for i, level in enumerate(self.layers[1:]):
            act = captum.attr.LayerActivation(self, level)
            act_vals = act.attribute(test_dataset, attribute_to_layer_input=True)
            
            cols = self.layer_info[i]
            if hasattr(self, 'data_index') : 
                data_index = self.data_index
            else : 
                data_index = np.arange(test_dataset.shape[0])
                
            act_vals_genomic = pd.DataFrame(act_vals.detach().cpu().numpy(),
                                            columns=cols,
                                            index=data_index)
            
            pathway_imp_by_target = act_vals_genomic
            layer_importance_scores.append(pathway_imp_by_target)
            
        self.interpret_flag=False
        return layer_importance_scores