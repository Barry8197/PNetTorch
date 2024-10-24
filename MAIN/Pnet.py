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
    """
    Applies a linear transformation to the incoming data followed by applying a mask: `y = (x @ (A * M)^T) + b`
    where `A` is the weights matrix, `M` is the mask, and `b` is the bias.

    This module supports specialized behaviors on different devices and data types to optimize performance.

    Parameters:
        mask (torch.Tensor): A 2D tensor that represents the mask to be applied to the weights.
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to `False`, the layer will not include an additive bias. Defaults to `True`.
        device (torch.device): The device to store the tensors (CPU or GPU). Defaults to None, which means use the current default device.
        dtype (torch.dtype): The desired data type of the parameters. Defaults to None, which means use the default data type.

    Shapes:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of additional dimensions and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension are the same shape as the input.

    Attributes:
        weight (torch.Tensor): Learnable weights of shape (out_features, in_features). These weights are modified by the mask.
        bias (torch.Tensor): Learnable bias of shape (out_features). Included only if bias is True.

    Example:
        >>> m = MaskedLinear(torch.ones(20, 30), 20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())  
        torch.Size([128, 30])
    """

    def __init__(self, mask: torch.Tensor, in_features: int, out_features: int, bias: bool=True, device=None, dtype=None):
        """
        Initializes the MaskedLinear layer with the given parameters, setting up parameters 
        such as the weight, bias, and applying the initial mask to the weights.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.mask = nn.Parameter(torch.Tensor(mask.T), requires_grad=False)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask_weights_init()

    def reset_parameters(self) -> None:
        """
        Resets the weights and biases to their initial states according to Kaiming uniform initialization,
        adjusting based on the presence of a bias.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def mask_weights_init(self):
        """
        Apply the mask to the weights upon initialization and normalize the weights to maintain 
        the distribution's scale post-masking.
        """
        self.weight.data = self.weight * self.mask
        non_zero_sum = self.mask.sum()
        if non_zero_sum != 0:
            scaling_factor = self.weight.data.sum() / non_zero_sum
            self.weight.data = self.mask * scaling_factor
        else:
            self.weight.data = self.mask 

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the module where the input tensor is linearly transformed,
        and the specified mask is applied element-wise to the weights during the transformation.
        
        Parameters:
            input (torch.Tensor): The input data.
        
        Returns:
            torch.Tensor: The transformed output.
        """
        return F.linear(input, self.weight * self.mask, self.bias)

    def extra_repr(self) -> str:
        """
        Sets the extra representation of the module, which will be displayed in the module representations.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

class PNET(nn.Module):
    """
    A PyTorch neural network module implementing a pathway network with optional feature importance 
    analyses using Captum.

    Args:
        reactome_network (ReactomeNetwork): A network structure providing masks for gene and pathway layers.
        input_dim (int): The number of input features.
        output_dim (int): The number of outputs.
        fcnn (bool): Flag to use a fully connected neural network, ignoring pathway structures.
        activation (torch.nn.Module): The activation function used in the neural network layers.
        dropout (float): The dropout rate.
        filter_pathways (bool): Flag to indicate whether to filter pathways based on some criterion in the network structure.
        input_layer_mask (torch.Tensor): An optional mask to be applied to the first layer.
    
    Attributes:
        layers (nn.ModuleList): List of linear or masked linear layers.
        skip (nn.ModuleList): List of layers used for skip connections in the network.
        layer_info (list): Information from the `reactome_network` about the layers.
    """
    def __init__(
        self, reactome_network, input_dim=None, output_dim=None, 
        fcnn=False, activation=nn.ReLU, dropout=0.1, 
        filter_pathways=False, input_layer_mask=None):
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
        
        self.layers = nn.ModuleList()
        self.skip = nn.ModuleList()
        
        if input_layer_mask is None:
            self.layers.append(nn.Linear(in_features=self.input_dim, out_features=gene_masks.shape[0]))
        else:
            self.layers.append(MaskedLinear(input_layer_mask, in_features=self.input_dim, out_features=gene_masks.shape[0]))

        for i in range(0, len(pathway_masks) + 1):
            if i == 0:
                self.layers.append(MaskedLinear(gene_masks, in_features=gene_masks.shape[0], out_features=pathway_masks[i].shape[0]))
                self.skip.append(nn.Linear(in_features=gene_masks.shape[0], out_features=self.output_dim))
            else:
                self.layers.append(MaskedLinear(pathway_masks[i-1], in_features=pathway_masks[i-1].shape[0], out_features=pathway_masks[i-1].shape[1]))
                self.skip.append(nn.Linear(in_features=pathway_masks[i-1].shape[0], out_features=self.output_dim))
                
        self.skip.append(nn.Linear(in_features=pathway_masks[-1].shape[1], out_features=self.output_dim))

    def forward(self, x):
        """
        Forward pass for the PNET model, applying each layer to the input sequentially,
        and adding the result of skip layers.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing through the PNET.
        """
        y = 0
        for layer, skip in zip(self.layers, self.skip):
            x = layer(x)
            y += skip(x)
            
        y = y / len(self.layers)
        
        return y

    def deepLIFT_feature_importance(self, test_dataset, target_class=0):
        """
        Calculate feature importances using the DeepLIFT algorithm through Captum.

        Args:
            test_dataset (torch.Tensor): The dataset for which to calculate importances.
            target_class (int): The target class index for which to calculate importances.

        Returns:
            pd.DataFrame: A dataframe containing the feature importances.
        """
        self.interpret_flag = True
        dl = captum.attr.DeepLift(self)
        feature_importances = dl.attribute(test_dataset, target=target_class)
        data_index = getattr(self, 'data_index', np.arange(test_dataset.shape[0]))

        feature_importances = pd.DataFrame(feature_importances.detach().cpu().numpy(),
                                           index=data_index,
                                           columns=self.features)
        
        self.feature_importances = feature_importances
        self.interpret_flag = False
        
        return self.feature_importances

    def layerwise_importance(self, test_dataset, target_class=0):
        """
        Compute layer-wise importance scores across all layers for given targets.

        Args:
            test_dataset (torch.Tensor): The dataset for which to calculate importances.
            target_class (int): The target class index for importance calculation.

        Returns:
            List[pd.DataFrame]: A list containing the importance scores for each layer.
        """
        self.interpret_flag = True
        layer_importance_scores = []
        
        for i, level in enumerate(self.layers):
            print(level)
            cond = captum.attr.LayerConductance(self, level)
            cond_vals = cond.attribute(test_dataset, target=target_class)
            cols = self.layer_info[i]
            data_index = getattr(self, 'data_index', np.arange(test_dataset.shape[0]))

            cond_vals_genomic = pd.DataFrame(cond_vals.detach().cpu().numpy(),
                                             columns=cols,
                                             index=data_index)
            layer_importance_scores.append(cond_vals_genomic)
            
        self.interpret_flag = False
        
        return layer_importance_scores