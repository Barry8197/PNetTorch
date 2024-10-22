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
    def __init__(self, reactome_network, input_dim =None ,  output_dim=None, fcnn=False):
        super().__init__()
        self.reactome_network = reactome_network
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
        gene_masks, pathway_masks = self.reactome_network.get_masks()
                
        if fcnn:
            gene_masks = [np.ones_like(gm) for gm in gene_masks]
            pathway_masks = [np.ones_like(gm) for gm in pathway_masks]
        
        # Prepare list of layers and list of predictions per layer:
        self.layers = nn.ModuleList()
        self.skip = nn.ModuleList()
        
        self.layers.append(nn.Linear(in_features = self.input_dim , out_features = gene_masks.shape[0]))
        
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
            x =  layer(x) # Initial Input Linear Layer
            y += skip(x)
            
        # Average over output
        y = y/(len(self.layers))
        
        return y

    def deepLIFT(self, test_dataset, target_class=0):
        self.interpret_flag=True
        dl = captum.attr.DeepLift(self)
        gene_importances = dl.attribute((test_dataset.x), target=target_class)
        gene_importances = pd.DataFrame(gene_importances.detach().numpy(),
                                        index=test_dataset.input_df.index,
                                        columns=test_dataset.input_df.columns)
        
        self.gene_importances = gene_importances
        self.interpret_flag=False
        return self.gene_importances
    
    def integrated_gradients(self, test_dataset, target_class=0 , task='Classification'):
        self.interpret_flag=True
        ig = captum.attr.IntegratedGradients(self)
        
        if task == 'REG':
            ig_attr = ig.attribute((test_dataset.x, test_dataset.additional), n_steps=50)
        else:
            ig_attr, delta = ig.attribute((test_dataset.x, test_dataset.additional), return_convergence_delta=True, target=target_class)
        gene_importances, additional_importances = ig_attr
        gene_importances = pd.DataFrame(gene_importances.detach().numpy(),
                                        index=test_dataset.input_df.index,
                                        columns=test_dataset.input_df.columns)
        additional_importances = pd.DataFrame(additional_importances.detach().numpy(),
                                              index=test_dataset.additional_data.index,
                                              columns=test_dataset.additional_data.columns)
        self.gene_importances, self.additional_importances = gene_importances, additional_importances
        self.interpret_flag=False
        return self.gene_importances, self.additional_importances

    def layerwise_importance(self, test_dataset, target_class=0):
        self.interpret_flag=True
        layer_importance_scores = []
        cond = captum.attr.LayerConductance(self, self.first_gene_layer)  # ReLU output of masked layer at each level
        cond_vals = cond.attribute((test_dataset.x, test_dataset.additional), target=target_class)
        cols = [self.reactome_network.pathway_encoding.set_index('ID').loc[col]['pathway'] for col in self.reactome_network.pathway_layers[0].index]
        cond_vals_genomic = pd.DataFrame(cond_vals.detach().numpy(),
                                         columns=cols,
                                         index=test_dataset.input_df.index)
        pathway_imp_by_target = cond_vals_genomic
        layer_importance_scores.append(pathway_imp_by_target)
        
        for i, level in enumerate(self.layers):
            cond = captum.attr.LayerConductance(self, level.pathway_layer)  # ReLU output of masked layer at each level
            cond_vals = cond.attribute((test_dataset.x, test_dataset.additional), target=target_class)
            cols = [self.reactome_network.pathway_encoding.set_index('ID').loc[col]['pathway'] for col in self.reactome_network.pathway_layers[i].columns]
            cond_vals_genomic = pd.DataFrame(cond_vals.detach().numpy(),
                                             columns=cols,
                                             index=test_dataset.input_df.index)
            pathway_imp_by_target = cond_vals_genomic
            layer_importance_scores.append(pathway_imp_by_target)
        self.interpret_flag=False
        return layer_importance_scores
    
    def layerwise_activation(self, test_dataset, target_class=0):
        self.interpret_flag=True
        layer_importance_scores = []
        for i, level in enumerate(self.layers):
            act = captum.attr.LayerActivation(self, level.pathway_layer)
            act_vals = act.attribute((test_dataset.x, test_dataset.additional), attribute_to_layer_input=True)
            cols = [self.reactome_network.pathway_encoding.set_index('ID').loc[col]['pathway'] for col in self.reactome_network.pathway_layers[i].index]
            act_vals_genomic = pd.DataFrame(act_vals.detach().numpy(),
                                            columns=cols,
                                            index=test_dataset.input_df.index)
            pathway_imp_by_target = act_vals_genomic
            layer_importance_scores.append(pathway_imp_by_target)
        self.interpret_flag=False
        return layer_importance_scores
    
    def neuron_conductance(self, test_dataset, target_class=0):
        self.interpret_flag=True
        layer_importance_scores = []
        for i, level in enumerate(self.layers):
            neuron_cond = captum.attr.NeuronConductance(self, level.pathway_layer)
            neuron_cond_att = neuron_cond.attribute((test_dataset.x, test_dataset.additional), target=target_class)
            
        self.interpret_flag=False    
    
    def gene_importance(self, test_dataset, target_class=0):
        self.interpret_flag=True
        cond = captum.attr.LayerConductance(self, self.input_layer)
        cond_vals = cond.attribute((test_dataset.x, test_dataset.additional), target=target_class)
        cols = self.reactome_network.gene_list
        cond_vals_genomic = pd.DataFrame(cond_vals.detach().numpy(),
                                         columns=cols,
                                         index=test_dataset.input_df.index)
        gene_imp_by_target = cond_vals_genomic
        self.interpret_flag=False
        return gene_imp_by_target
    
    def regulatory_layer_importance(self, test_dataset, target_class=0):
        self.interpret_flag=True
        cond = captum.attr.LayerConductance(self, self.regulatory_layer.regulatory_layer)
        cond_vals = cond.attribute((test_dataset.x, test_dataset.additional), target=target_class)
        cols = self.reactome_network.gene_list
        cond_vals_genomic = pd.DataFrame(cond_vals.detach().numpy(),
                                         columns=cols,
                                         index=test_dataset.input_df.index)
        gene_imp_by_target = cond_vals_genomic
        self.interpret_flag=False
        return gene_imp_by_target
    
    def interpret(self, test_dataset, plot=False):
        gene_feature_importances, additional_feature_importances = self.integrated_gradients(test_dataset)
        gene_importances = self.gene_importance(test_dataset)
        # layer_importance_scores = self.layerwise_importance(test_dataset)
        if self.regulatory_flag == True:
            regulatory_importances = self.regulatory_layer_importance(test_dataset)

        layer_importance_scores = self.layerwise_importance(test_dataset)
        
        gene_order = gene_importances.mean().sort_values(ascending=True).index
        if plot:
            plt.rcParams["figure.figsize"] = (6,8)
            gene_importances[list(gene_order[-20:])].plot(kind='box', vert=False)
            plt.savefig(plot+'/imp_genes.pdf')
        self.interpret_flag=False
        if self.regulatory_flag == True:
            return gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores, regulatory_importances
        else: 
            return gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores,

    def interpret(self, test_dataset, plot=False):
        gene_feature_importances, additional_feature_importances = self.integrated_gradients(test_dataset)
        gene_importances = self.gene_importance(test_dataset)
        # layer_importance_scores = self.layerwise_importance(test_dataset)
        layer_importance_scores = self.layerwise_importance(test_dataset)
        
        gene_order = gene_importances.mean().sort_values(ascending=True).index
        if plot:
            plt.rcParams["figure.figsize"] = (6,8)
            gene_importances[list(gene_order[-20:])].plot(kind='box', vert=False)
            plt.savefig(plot+'/imp_genes.pdf')
        self.interpret_flag=False            
        return gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores

def evaluate_interpret_save(model, test_dataset, path):
    if not os.path.exists(path):
        os.makedirs(path)
    x_test = test_dataset.x
    additional_test = test_dataset.additional
    y_test = test_dataset.y
    model.to('cpu')
    if model.task=='BC' or model.task=='MC':
        pred_proba = model.predict_proba(x_test, additional_test).detach()
        pred = model.predict(x_test, additional_test).detach()
        auc_score = util.get_auc(pred_proba, y_test, save=path+'/auc_curve.pdf')
        auc_prc = util.get_auc_prc(pred_proba, y_test)
        f1_score = util.get_f1(pred, y_test)
        
    
        torch.save(pred_proba, path+'/prediction_probabilities.pt')
        torch.save(auc_score, path+'/AUC.pt')
        torch.save(auc_prc, path+'/AUC_PRC.pt')
        torch.save(f1_score, path+'/F1.pt')
        
    gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores = model.interpret(test_dataset)
    gene_feature_importances.to_csv(path+'/gene_feature_importances.csv')
    additional_feature_importances.to_csv(path+'/additional_feature_importances.csv')
    gene_importances.to_csv(path+'/gene_importances.csv')
    for i, layer in enumerate(layer_importance_scores):
        layer.to_csv(path+'/layer_{}_importances.csv'.format(i))

def interpret(model, x, additional,  plots=False, savedir=''):
    '''
    Function to use DeepLift from Captum on PNET model structure. Generates overall feature importance and layerwise
    results.
    :param model: NN model to predict feature importance on. Assuming PNET structure
    :param data: PnetDataset; data object with samples to use gradients on.
    :return:
    '''
    if plots:
        if savedir:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
        else:
            savedir = os.getcwd()
    feature_importance = dict()
    # Overall feature importance
    ig = IntegratedGradients(model)
    ig_attr, delta = ig.attribute((x, additional), return_convergence_delta=True)
    ig_attr_genomic, ig_attr_additional = ig_attr
    feature_importance['overall_genomic'] = ig_attr_genomic.detach().numpy()
    feature_importance['overall_clinical'] = ig_attr_additional.detach().numpy()
    if plots:
        visualize_importances(test_df.columns[:clinical_index],
                              np.mean(feature_importance['overall_clinical'], axis=0),
                              title="Average Feature Importances",
                              axis_title="Clinical Features")
        plt.savefig('/'.join([ savedir, 'feature_importance_overall_clinical.pdf']))

        visualize_importances(test_df.columns[clinical_index:],
                              np.mean(feature_importance['overall_genomic'], axis=0),
                              title="Average Feature Importances",
                              axis_title="Genomic Features")
        plt.savefig('/'.join([savedir, 'feature_importance_overall_genomic.pdf']))

    # Neurons feature importance
    layer_importance_scores = []
    for level in model.layers:
        cond = LayerConductance(model, level.activation)       # ReLU output of masked layer at each level
        cond_vals = cond.attribute((genomic_input, clinical_input))
        cond_vals_genomic = cond_vals.detach().numpy()
        layer_importance_scores.append(cond_vals_genomic)
    feature_importance['layerwise_neurons_genomic'] = layer_importance_scores
    if plots:
        for i, layer in enumerate(feature_importance['layerwise_neurons_genomic']):
            pathway_names = model.reactome_network.pathway_encoding.set_index('ID')
            pathway_names = pathway_names.loc[model.reactome_network.pathway_layers[i+1].index]['pathway']
            visualize_importances(pathway_names,
                                  np.mean(layer, axis=0),
                                  title="Neurons Feature Importances",
                                  axis_title="Pathway activation Features")
            plt.savefig('/'.join([savedir, 'pathway_neurons_layer_{}_importance.pdf'.format(i)]))

    return feature_importance


def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, rotation=90)
        plt.xlabel(axis_title)
        plt.title(title)
