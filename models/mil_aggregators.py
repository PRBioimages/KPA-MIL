# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
from .model_utils import *
import math
import warnings
import sys
import os
import contextlib

# Add MIL_BASELINE-main12.9 to path
MIL_BASELINE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MIL_BASELINE-main12.9')
sys.path.insert(0, MIL_BASELINE_PATH)

@contextlib.contextmanager
def mil_baseline_path_context():
    """Context manager to temporarily set working directory and path for MIL_BASELINE imports"""
    original_cwd = os.getcwd()
    original_path = sys.path.copy()
    
    try:
        # Change to MIL_BASELINE directory
        os.chdir(MIL_BASELINE_PATH)
        # Ensure MIL_BASELINE is in the path
        if MIL_BASELINE_PATH not in sys.path:
            sys.path.insert(0, MIL_BASELINE_PATH)
        yield
    finally:
        # Restore original state
        os.chdir(original_cwd)
        sys.path = original_path


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ConsistencyLoss(nn.Module):
    def __init__(self, temperature=0.1, noise_std=0.1):
        super().__init__()
        self.temperature = temperature
        self.noise_std = noise_std
    
    def forward(self, features, noisy_features=None):
        features = F.normalize(features, dim=1)
        if noisy_features is None:
            noise = torch.randn_like(features) * self.noise_std
            noisy_features = features + noise
        noisy_features = F.normalize(noisy_features, dim=1)
        loss = -torch.mean(torch.sum(features * noisy_features, dim=1))
        return loss


class CrossScaleContrastiveLoss(nn.Module):
    """
    Cross-scale contrastive loss for aligning low and high resolution features.
    
    Design principles:
    1. Intra-scale contrast (always active): low and high resolution features of the same image should be similar
    2. Inter-scale contrast (only when batch_size > 1): features from different images should be dissimilar
    
    Args:
        temperature: Temperature parameter for contrastive loss (default: 0.5)
        weight: Overall weight of the contrastive loss (default: 1.0)
        intra_weight: Weight for intra-image contrast (MSE loss) (default: 1.0)
        inter_weight: Weight for inter-image contrast (InfoNCE) (default: 0.5)
    """
    def __init__(self, temperature=0.5, weight=1.0, 
                 intra_weight=1.0, inter_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        
    def forward(self, features_low, features_high):
        """
        Args:
            features_low: [batch_size, feat_dim] - Low resolution global features
            features_high: [batch_size, feat_dim] - High resolution global features
        Returns:
            loss: Cross-scale contrastive loss
        """
        features_low = F.normalize(features_low, dim=1)
        features_high = F.normalize(features_high, dim=1)
        
        batch_size = features_low.size(0)
        
        # ========== 1. Intra-image contrast (always active) ==========
        # MSE loss to pull low and high resolution features of the same image closer
        intra_loss = F.mse_loss(features_low, features_high)
        total_loss = self.intra_weight * intra_loss
        
        # ========== 2. Inter-image contrast (only when batch_size > 1) ==========
        if batch_size > 1:
            # Concatenate all features: [low_1, low_2, ..., high_1, high_2, ...]
            all_features = torch.cat([features_low, features_high], dim=0)
            
            # Compute similarity matrix
            sim_matrix = all_features @ all_features.T / self.temperature
            
            # Positive pairs: low_i pairs with high_i
            labels = torch.arange(batch_size, device=features_low.device)
            labels = torch.cat([labels, labels], dim=0)  # [0,1,2,...,0,1,2,...]
            
            # NT-Xent loss
            inter_loss = F.cross_entropy(sim_matrix, labels)
            total_loss = total_loss + self.inter_weight * inter_loss
        
        return self.weight * total_loss


class MILAggregator(nn.Module):
    """
    Base class for MIL aggregation methods
    """
    def __init__(self, config):
        super(MILAggregator, self).__init__()
        self.config = config
        self.L = config.input_size  # Feature dimension
        self.D = config.hidden_size  # Hidden dimension
        self.K = 1
    
    def forward(self, x_s, x_l):
        """
        Forward pass for MIL aggregation
        
        Args:
            x_s: Low resolution patch features
            x_l: High resolution patch features
            
        Returns:
            image_features_low: Aggregated low resolution features
            image_features_high: Aggregated high resolution features
            compents_low: Low resolution components (for text context)
            compents_high: High resolution components (for text context)
            M_low: Original low resolution features
            M_high: Original high resolution features
        """
        raise NotImplementedError("Subclasses must implement forward method")


class AMILAggregator(MILAggregator):
    """
    Traditional Attention-based MIL Aggregator
    """
    def __init__(self, config):
        super(AMILAggregator, self).__init__(config)
        
        # Attention mechanism for MIL
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
    
    def forward(self, x_s, x_l):
        # Get device from model parameters
        device = next(self.parameters()).device if hasattr(self, 'attention_V') else torch.device('cpu')
        
        # Move input tensors to the correct device
        x_s = x_s.to(device)
        x_l = x_l.to(device)
        
        # Low resolution processing
        M_low = x_s.float()
        
        # High resolution processing
        M_high = x_l.float()
        
        # Attention pooling for low resolution
        H_low = M_low.float()
        A_V_low = self.attention_V(H_low)
        A_U_low = self.attention_U(H_low)
        A_low = self.attention_weights(A_V_low * A_U_low)
        A_low = torch.transpose(A_low, 1, 0)
        A_low = F.softmax(A_low, dim=1)
        image_features_low = torch.mm(A_low, H_low)
        
        # Attention pooling for high resolution
        H_high = M_high.float()
        A_V_high = self.attention_V(H_high)
        A_U_high = self.attention_U(H_high)
        A_high = self.attention_weights(A_V_high * A_U_high)
        A_high = torch.transpose(A_high, 1, 0)
        A_high = F.softmax(A_high, dim=1)
        image_features_high = torch.mm(A_high, H_high)
        
        # Return empty tensors for compents since AMIL doesn't use prototype enhancement
        # The model will handle empty compents by directly using M_low/M_high
        device = M_low.device
        compents_low = torch.empty(0, self.L, device=device)
        compents_high = torch.empty(0, self.L, device=device)
        
        return image_features_low, image_features_high, compents_low, compents_high, M_low, M_high


class PrototypeMILAggregator(MILAggregator):
    """
    Prototype-guided MIL Aggregator with cross-scale contrastive learning support
    """
    def __init__(self, config):
        super(PrototypeMILAggregator, self).__init__(config)
        
        # Attention mechanism for MIL
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
        
        # Shared prototype-guided patch decoder
        self.norm = nn.LayerNorm(config.input_size)
        num_heads = getattr(config, 'num_heads', 1)
        prototype_init_std = getattr(config, 'prototype_init_std', 0.02)
        self.cross_attention_1 = MultiheadAttention(embed_dim=config.input_size, num_heads=num_heads)
        self.learnable_image_center = nn.Parameter(torch.Tensor(*[config.prototype_number, 1, config.input_size]))
        trunc_normal_(self.learnable_image_center, std=prototype_init_std)
        
        # Cross-scale contrastive learning is now handled at the model level
        # This flag is kept for compatibility but not used here
    
    def forward(self, x_s, x_l):
        # Get device from model parameters
        device = next(self.parameters()).device if hasattr(self, 'attention_V') else torch.device('cpu')
        
        # Move input tensors to the correct device
        x_s = x_s.to(device)
        x_l = x_l.to(device)
        
        # Low resolution processing
        M_low = x_s.float()
        compents_low, _ = self.cross_attention_1(self.learnable_image_center, M_low, M_low)
        compents_low = self.norm(compents_low + self.learnable_image_center)
        
        # High resolution processing
        M_high = x_l.float()
        compents_high, _ = self.cross_attention_1(self.learnable_image_center, M_high, M_high)
        compents_high = self.norm(compents_high + self.learnable_image_center)
        
        # Attention pooling for low resolution
        H_low = compents_low.squeeze().float()
        A_V_low = self.attention_V(H_low)
        A_U_low = self.attention_U(H_low)
        A_low = self.attention_weights(A_V_low * A_U_low)
        A_low = torch.transpose(A_low, 1, 0)
        A_low = F.softmax(A_low, dim=1)
        image_features_low = torch.mm(A_low, H_low)
        
        # Attention pooling for high resolution
        H_high = compents_high.squeeze().float()
        A_V_high = self.attention_V(H_high)
        A_U_high = self.attention_U(H_high)
        A_high = self.attention_weights(A_V_high * A_U_high)
        A_high = torch.transpose(A_high, 1, 0)
        A_high = F.softmax(A_high, dim=1)
        image_features_high = torch.mm(A_high, H_high)
        
        return image_features_low, image_features_high, compents_low, compents_high, M_low, M_high


class BaselineMILAggregator(MILAggregator):
    """
    MIL Aggregator that wraps MIL methods from MIL_BASELINE-main12.9
    """
    def __init__(self, config):
        super(BaselineMILAggregator, self).__init__(config)
        
        # Get MIL method name from config
        self.mil_method = getattr(config, 'mil_method', 'AB_MIL')
        
        # Get MIL configuration if available
        mil_config = getattr(config, 'mil_config', None)
        
        # Prepare model parameters
        model_params = {}
        
        # Use mil_config if available, otherwise use default values
        if mil_config is not None:
            model_params.update(mil_config)
        
        # Set default values if not in mil_config
        model_params.setdefault('in_dim', config.input_size)
        model_params.setdefault('num_classes', 1)  # We'll handle multi-class later
        model_params.setdefault('L', config.input_size)
        model_params.setdefault('D', config.hidden_size)
        model_params.setdefault('dropout', 0.1)
        model_params.setdefault('act', nn.ReLU())
        
        # Convert string activation to function if needed
        if isinstance(model_params['act'], str):
            act_str = model_params['act'].lower()
            if act_str == 'relu':
                model_params['act'] = nn.ReLU()
            elif act_str == 'silu':
                model_params['act'] = nn.SiLU()
            elif act_str == 'gelu':
                model_params['act'] = nn.GELU()
            else:
                model_params['act'] = nn.ReLU()
        
        # Import the corresponding MIL module
        try:
            # First check if it's a simple method like AB_MIL that doesn't have dependencies
            if self.mil_method == 'AB_MIL':
                from modules.AB_MIL.ab_mil import AB_MIL
                self.mil_model = AB_MIL(**model_params)
                print(f"Loaded MIL method: {self.mil_method} with parameters: {model_params}")
            elif self.mil_method == 'DTFD_MIL':
                # DTFD_MIL is composed of multiple components
                from modules.DTFD_MIL.dtfd_mil import Classifier_1fc, Attention, DimReduction, Attention_with_Classifier
                
                # Set default parameters for DTFD_MIL
                mdim = model_params.get('mdim', 512)
                numLayer_Res = model_params.get('numLayer_Res', 0)
                classifier_dropout = model_params.get('classifier_dropout', 0.1)
                attCls_dropout = model_params.get('attCls_dropout', 0.1)
                num_Group = model_params.get('num_Group', 4)  # Number of groups
                distill = model_params.get('distill', 'MaxMinS')  # Distillation strategy
                total_instance = model_params.get('total_instance', 100)  # Total instances
                instance_per_group = total_instance // num_Group
                
                # Create DTFD_MIL components
                self.classifier = Classifier_1fc(mdim, model_params['num_classes'], classifier_dropout)
                self.attention = Attention(mdim)
                self.dimReduction = DimReduction(model_params['in_dim'], mdim, numLayer_Res=numLayer_Res)
                self.attCls = Attention_with_Classifier(L=mdim, num_cls=model_params['num_classes'], droprate=attCls_dropout)
                
                # Create a wrapper class for DTFD_MIL that matches original implementation
                class DTFD_MIL_Wrapper(nn.Module):
                    def __init__(self, classifier, attention, dimReduction, attCls, num_Group, distill, instance_per_group):
                        super(DTFD_MIL_Wrapper, self).__init__()
                        self.classifier = classifier
                        self.attention = attention
                        self.dimReduction = dimReduction
                        self.attCls = attCls
                        self.num_Group = num_Group
                        self.distill = distill
                        self.instance_per_group = instance_per_group
                    
                    def get_cam_1d(self, classifier, features):
                        """Calculate CAM for instance selection"""
                        tweight = list(classifier.parameters())[-2]
                        cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
                        return cam_maps
                    
                    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
                        forward_return = {}
                        
                        # Process input - handle different input shapes
                        if x.dim() == 3:
                            x = x.squeeze(0)  # Remove batch dimension if present
                        
                        # Split bag into groups
                        inputs_pseudo_bags = torch.chunk(x, self.num_Group, dim=0)
                        
                        slide_sub_preds = []
                        slide_pseudo_feat = []
                        
                        for subFeat_tensor in inputs_pseudo_bags:
                            # Reduce dimension
                            tmidFeat = self.dimReduction(subFeat_tensor)
                            
                            # Get attention weights
                            tAA = self.attention(tmidFeat).squeeze(0)
                            
                            # Compute weighted features using einsum
                            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)
                            
                            # Aggregate features
                            tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)
                            
                            # Get prediction from classifier
                            tPredict = self.classifier(tattFeat_tensor)
                            
                            # Instance selection using CAM
                            tattFeats_expanded = tattFeats.unsqueeze(0)
                            patch_pred_logits = self.get_cam_1d(self.classifier, tattFeats_expanded).squeeze(0)
                            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)
                            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)
                            
                            # Select top-k and bottom-k instances based on last column
                            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
                            topk_idx_max = sort_idx[:self.instance_per_group].long()
                            topk_idx_min = sort_idx[-self.instance_per_group:].long()
                            
                            # Select features based on distillation strategy
                            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                            
                            if self.distill == 'MaxMinS':
                                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                                MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                                slide_pseudo_feat.append(MaxMin_inst_feat)
                            elif self.distill == 'MaxS':
                                slide_pseudo_feat.append(max_inst_feat)
                            elif self.distill == 'AFS':
                                slide_pseudo_feat.append(tattFeat_tensor)
                            
                            slide_sub_preds.append(tPredict)
                        
                        # Concatenate all pseudo features
                        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
                        
                        # Final classification using attCls
                        forward_return = self.attCls(slide_pseudo_feat, return_WSI_attn=return_WSI_attn, return_WSI_feature=return_WSI_feature)
                        
                        return forward_return
                
                # Create wrapped model
                self.mil_model = DTFD_MIL_Wrapper(
                    self.classifier, self.attention, self.dimReduction, self.attCls,
                    num_Group, distill, instance_per_group
                )
                print(f"Loaded MIL method: {self.mil_method} with parameters: {model_params}")
            else:
                # For other methods, try to import normally
                module_name = f"modules.{self.mil_method}.{self.mil_method.lower()}"
                mil_module = __import__(module_name, fromlist=['*'])
                
                # Try to get the class
                try:
                    mil_model_class = getattr(mil_module, self.mil_method)
                    self.mil_model = mil_model_class(**model_params)
                    print(f"Loaded MIL method: {self.mil_method} with parameters: {model_params}")
                except AttributeError:
                    # If the class doesn't exist, try to find a suitable class
                    module_classes = [cls for cls in dir(mil_module) 
                                    if isinstance(getattr(mil_module, cls), type) 
                                    and issubclass(getattr(mil_module, cls), nn.Module)]
                    if module_classes:
                        mil_model_class = getattr(mil_module, module_classes[0])
                        self.mil_model = mil_model_class(**model_params)
                        print(f"Loaded MIL method: {self.mil_method} using class {module_classes[0]} with parameters: {model_params}")
                    else:
                        raise ValueError(f"No suitable model class found in module {module_name}")
        except Exception as e:
            raise ValueError(f"Failed to load MIL method {self.mil_method}: {str(e)}")
    
    def forward(self, x_s, x_l):
        # Get device from model parameters
        device = next(self.parameters()).device if hasattr(self, 'mil_model') else torch.device('cpu')
        
        # Move input tensors to the correct device
        x_s = x_s.to(device)
        x_l = x_l.to(device)
        
        # Low resolution processing
        M_low = x_s.float()
        # Add batch dimension if missing
        if M_low.dim() == 2:
            M_low = M_low.unsqueeze(0)
        
        # High resolution processing
        M_high = x_l.float()
        # Add batch dimension if missing
        if M_high.dim() == 2:
            M_high = M_high.unsqueeze(0)
        
        # Use baseline MIL model for aggregation
        try:
            # Process low resolution features
            output_low = self.mil_model(M_low, return_WSI_feature=True)
            image_features_low = output_low.get('WSI_feature', None)
            if image_features_low is None:
                # Fallback: use logits if feature is not available
                image_features_low = output_low.get('logits', None)
            
            # Process high resolution features
            output_high = self.mil_model(M_high, return_WSI_feature=True)
            image_features_high = output_high.get('WSI_feature', None)
            if image_features_high is None:
                # Fallback: use logits if feature is not available
                image_features_high = output_high.get('logits', None)
            
            # Ensure features have correct shape
            if image_features_low is not None:
                if image_features_low.dim() == 0:
                    image_features_low = image_features_low.unsqueeze(0)
                elif image_features_low.dim() == 3:
                    image_features_low = image_features_low.squeeze(0)
                elif image_features_low.dim() == 1:
                    # Add batch dimension if needed
                    image_features_low = image_features_low.unsqueeze(0)
            
            if image_features_high is not None:
                if image_features_high.dim() == 0:
                    image_features_high = image_features_high.unsqueeze(0)
                elif image_features_high.dim() == 3:
                    image_features_high = image_features_high.squeeze(0)
                elif image_features_high.dim() == 1:
                    # Add batch dimension if needed
                    image_features_high = image_features_high.unsqueeze(0)
                
            
        except Exception as e:
            # Fallback to simple mean pooling if baseline MIL fails
            print(f"Warning: Baseline MIL failed, using mean pooling fallback: {str(e)}")
            image_features_low = torch.mean(M_low, dim=0, keepdim=True)
            image_features_high = torch.mean(M_high, dim=0, keepdim=True)
        
        # For compatibility with prototype-based implementation
        # compents needs to have shape [1, num_patches, input_size] for .squeeze() to work correctly
        # Then compents.squeeze() will give [num_patches, input_size] which can be concatenated with M_low/M_high
        if M_low.dim() == 2:
            compents_low = M_low.unsqueeze(0)  # [1, num_patches, input_size]
        else:
            compents_low = M_low
        
        if M_high.dim() == 2:
            compents_high = M_high.unsqueeze(0)  # [1, num_patches, input_size]
        else:
            compents_high = M_high
        
        return image_features_low, image_features_high, compents_low, compents_high, M_low.squeeze(), M_high.squeeze()


def create_mil_aggregator(config):
    """
    Factory function to create MIL aggregator based on config
    
    Args:
        config: Configuration object with mil_aggregator parameter
        
    Returns:
        MILAggregator instance
    """
    mil_aggregator = getattr(config, 'mil_aggregator', 'amil')
    
    if mil_aggregator == 'amil':
        return AMILAggregator(config)
    elif mil_aggregator == 'prototype':
        return PrototypeMILAggregator(config)
    elif mil_aggregator == 'baseline':
        return BaselineMILAggregator(config)
    else:
        raise ValueError(f"Unknown MIL aggregator type: {mil_aggregator}")
