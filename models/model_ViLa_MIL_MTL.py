# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from os.path import join as pjoin
from .model_utils import *
from .mil_aggregators import create_mil_aggregator
from .mdn import SurvivalMDN, safe_inverse_softplus, inverse_softplus_grad, inverse_softplus
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import warnings

# 延迟导入 CLIP 模块，只在需要时导入
clip = None
_Tokenizer = None
_tokenizer = None


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class ConchTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load Conch model and tokenizer
        import sys
        import os
        # Add CONCH directory to path
        conch_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'CONCH')
        sys.path.insert(0, conch_dir)
        
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        
        # Use the Conch checkpoint path from config
        conch_checkpoint = getattr(config, 'conch_checkpoint', '/home/hzxu/CONCH/checkpoints/CONCH/')
        
        # Load model and preprocess
        # Assuming the model is in the conch subdirectory
        checkpoint_path = os.path.join(conch_checkpoint, 'pytorch_model.bin')
        self.model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
        
        # Get tokenizer
        self.tokenizer = get_tokenizer()
       
        # Get model's output dimension
        self.hidden_size = self.model.text.output_dim

        # Projection layer to match expected output size
        self.projection = nn.Linear(self.hidden_size, config.input_size)
        
        # Freeze/unfreeze configuration
        self.freeze_conch = getattr(config, 'freeze_conch', True)
        self.conch_unfreeze_layers = getattr(config, 'conch_unfreeze_layers', 1)
        
        # Apply freezing/unfreezing
        self._apply_freeze()
        
    def _apply_freeze(self):
        """Apply freezing to Conch text encoder based on configuration"""
        if self.freeze_conch:
            # Freeze all Conch parameters
            for param in self.model.parameters():
                param.requires_grad = False
            print("Conch text encoder is frozen")
        else:
            # Unfreeze the last N layers of Conch text encoder
            # First freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Then unfreeze the last N layers of the text transformer
            text_transformer = self.model.text.transformer
            if hasattr(text_transformer, 'resblocks'):
                # Get all layers
                layers = text_transformer.resblocks
                num_layers = len(layers)
                
                # Unfreeze the last N layers
                start_layer = max(0, num_layers - self.conch_unfreeze_layers)
                for i in range(start_layer, num_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                
                print(f"Conch text encoder: unfreezing last {self.conch_unfreeze_layers} layers (layers {start_layer} to {num_layers-1})")
        
        # Projection layer is always trainable
        for param in self.projection.parameters():
            param.requires_grad = True
        
    def forward(self, prompts, tokenized_prompts=None):
       
        # Import tokenize function from custom_tokenizer
        from conch.open_clip_custom import tokenize
        # Tokenize prompts using the provided tokenize function
        tokenized = tokenize(self.tokenizer, prompts)
        
        # Move tokenized inputs to same device as model
        model_device = next(self.model.parameters()).device
        tokenized = tokenized.to(model_device)
        
        # Get embeddings from Conch model
        # Only use no_grad if Conch is frozen
        if self.freeze_conch:
            with torch.no_grad():
                embeddings = self.model.encode_text(tokenized, normalize=False)
        else:
            embeddings = self.model.encode_text(tokenized, normalize=False)
        
        # Move embeddings to projection layer's device
        projection_device = next(self.projection.parameters()).device
        embeddings = embeddings.to(projection_device)
        # Project to match expected output size
        embeddings = self.projection(embeddings)
        
        return embeddings
    

class RandomConchTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Assuming Conch model is loaded similarly to CLIP
        # This is a placeholder implementation
        # You may need to adjust based on the actual Conch model API
        self.hidden_size = config.input_size
        # Dummy projection layer to match expected output size
        self.projection = nn.Linear(self.hidden_size, config.input_size)
        
    def forward(self, prompts, tokenized_prompts=None):
        # In a real implementation, this would use the Conch model
        # to encode the text prompts
        # For now, we'll generate random embeddings as a placeholder
        batch_size = len(prompts)
        # Get device from model parameters instead of prompts
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        # Generate random embeddings with the same size as input
        # embeddings = torch.randn(batch_size, self.hidden_size, device=prompts.device)
        embeddings = torch.randn(batch_size, self.hidden_size, device=device)
        # Project to match expected output size
        embeddings = self.projection(embeddings)
        return embeddings


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model=None, text_extractor='clip', config=None):
        super().__init__()
        self.text_extractor = text_extractor
        n_cls = len(classnames)
        
        # Get configurable parameters with defaults
        if config is not None:
            n_ctx = getattr(config, 'n_ctx', 16)
            ctx_init_std = getattr(config, 'ctx_init_std', 0.02)
            class_token_position = getattr(config, 'class_token_position', 'end')
        else:
            n_ctx = 16
            ctx_init_std = 0.02
            class_token_position = 'end'
        
        ctx_init = ""
        
        if text_extractor == 'clip':
            # 确保 CLIP 模块已导入
            global clip, _Tokenizer, _tokenizer
            if clip is None:
                import clip
                from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
                _tokenizer = _Tokenizer()
            
            dtype = clip_model.dtype
            ctx_dim = clip_model.ln_final.weight.shape[0]
            clip_imsize = clip_model.visual.input_resolution

            if ctx_init:
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = len(ctx_init.split(" "))
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                prompt_prefix = ctx_init

            else:
                if False:
                    ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                else:
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=ctx_init_std)
                prompt_prefix = " ".join(["X"] * n_ctx)

            self.ctx = nn.Parameter(ctx_vectors)  
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [name for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            self.register_buffer("token_prefix", embedding[:, :1, :])  
            self.register_buffer("token_suffix", embedding[:, 1:-n_ctx, :]) 

            self.n_cls = n_cls
            self.n_ctx = n_ctx
            self.tokenized_prompts = tokenized_prompts 
            self.name_lens = name_lens
            self.class_token_position = class_token_position
        else:  # conch
            # For Conch, we'll keep it simple
            self.prompts = list(classnames)

    def forward(self):
        if self.text_extractor == 'clip':
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix

            if self.class_token_position == "end":
                prompts = torch.cat(
                    [
                        prefix,  
                        ctx,     
                        suffix, 
                    ],
                    dim=1,
                )

            elif self.class_token_position == "middle":
                half_n_ctx = self.n_ctx // 2
                prompts = []
                for i in range(self.n_cls):
                    name_len = self.name_lens[i]
                    prefix_i = prefix[i : i + 1, :, :]
                    class_i = suffix[i : i + 1, :name_len, :]
                    suffix_i = suffix[i : i + 1, name_len:, :]
                    ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                    ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                    prompt = torch.cat(
                        [
                            prefix_i,     
                            ctx_i_half1,  
                            class_i,      
                            ctx_i_half2,  
                            suffix_i,   
                        ],
                        dim=1,
                    )
                    prompts.append(prompt)
                prompts = torch.cat(prompts, dim=0)

            elif self.class_token_position == "front":
                prompts = []
                for i in range(self.n_cls):
                    name_len = self.name_lens[i]
                    prefix_i = prefix[i : i + 1, :, :]
                    class_i = suffix[i : i + 1, :name_len, :]
                    suffix_i = suffix[i : i + 1, name_len:, :]
                    ctx_i = ctx[i : i + 1, :, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  
                            class_i,   
                            ctx_i,    # (1, *, dim)
                        ],
                        dim=1,
                    )
                    prompts.append(prompt)
                prompts = torch.cat(prompts, dim=0)
            else:
                raise ValueError
            return prompts
        else:  # conch
            # For Conch, return the prompts as is
            return self.prompts


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


class CoxPHLoss(nn.Module):
    """Cox proportional hazards loss for survival analysis"""
    def __init__(self):
        super(CoxPHLoss, self).__init__()
    
    def forward(self, risk_scores, durations, events):
        """
        risk_scores: predicted risk scores (higher means higher risk)
        durations: survival time
        events: 1 if event occurred, 0 if censored
        """

        # Handle scalar input (batch_size=1)
        if risk_scores.dim() == 0:
            risk_scores = risk_scores.unsqueeze(0)
        if durations.dim() == 0:
            durations = durations.unsqueeze(0)
        if events.dim() == 0:
            events = events.unsqueeze(0)

        # Ensure all tensors are on the same device
        device = risk_scores.device
        durations = durations.to(device)
        events = events.to(device)

        # Filter out invalid samples (NaN or inf)
        valid_mask = torch.isfinite(risk_scores) & torch.isfinite(durations) & torch.isfinite(events)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
        
        risk_scores = risk_scores[valid_mask]
        durations = durations[valid_mask]
        events = events[valid_mask]
        
        if events.sum() == 0:
            return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
        
        # Sort by duration in descending order
        idx = torch.argsort(durations, descending=True)
        risk_scores = risk_scores[idx]
        events = events[idx]
        durations = durations[idx]
        
        # Calculate log partial likelihood (Breslow approximation for ties)
        # Risk set for sample i: all samples with duration >= t_i
        cumsum = torch.cumsum(torch.exp(risk_scores), dim=0)
        # For tied durations, use the cumsum at the last position of each tied group
        unique_durs, unique_inverse = torch.unique(durations, return_inverse=True)
        n_groups = len(unique_durs)
        group_last_idx = torch.zeros(n_groups, dtype=torch.long, device=durations.device)
        for g in range(n_groups):
            positions = torch.where(unique_inverse == g)[0]
            group_last_idx[g] = positions[-1]
        risk_set = cumsum[group_last_idx[unique_inverse]]
        # Avoid log(0)
        risk_set = torch.clamp(risk_set, min=1e-8)
        diff = risk_scores - torch.log(risk_set)
        loss = -torch.sum(events * diff) / (torch.sum(events) + 1e-8)
        
        return loss


class MDNLoss(nn.Module):
    """Mixture Density Network loss for survival analysis
    Adapted from SCMIL project
    """
    def __init__(self):
        super(MDNLoss, self).__init__()
    
    def forward(self, datas):
        """
        datas: dict containing 'c' (censorship), 'pdf' (probability density function), 
              'survival_func' (survival function), 't' (survival time)
        """
        c = datas['c']
        pdf = datas['pdf']
        survival_func = datas['survival_func']
        t = datas['t'] if 't' in datas else None
        
        # Ensure all tensors are on the same device
        device = pdf.device
        c = c.to(device)
        survival_func = survival_func.to(device)
        if t is not None:
            t = t.to(device)
        
        # Limit maximum pdf value to prevent model from predicting all mass at t=0
        pdf = torch.clamp(pdf, max=0.99)
        
        # Original loss
        loss = -(1-c)*torch.log((pdf+1e-6)/(survival_func+1e-6)) - torch.log(survival_func + 1e-6)
        
        # Add penalty for predictions that are too concentrated at early times
        # This helps when all events are uncensored (event=1 for all)
        if t is not None and (1 - c).sum() > 0:
            event_mask = (1 - c) > 0
            if event_mask.sum() > 0:
                t_event = t[event_mask]
                # Penalize very small predicted times
                # The penalty increases as t approaches 0
                time_penalty = torch.clamp(1 - t_event, min=0).mean() * 0.1
                loss = loss + time_penalty
        
        return loss.mean()


class OrdinalLoss(nn.Module):
    """Ordinal loss for ordinal classification tasks"""
    def __init__(self, num_classes):
        super(OrdinalLoss, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, logits, labels):
        """
        logits: predicted logits (batch_size, num_classes)
        labels: ground truth labels (batch_size,)
        """
        # Ensure labels are on the same device as logits
        labels = labels.to(logits.device)
        
        # Convert logits to cumulative probabilities using sigmoid
        # For ordinal classification, we use cumulative logits
        # logits shape: (batch_size, num_classes-1)
        if logits.shape[1] == self.num_classes:
            # If logits have num_classes channels, use the first num_classes-1
            logits = logits[:, :-1]
        
        # Create target matrix for ordinal loss
        batch_size = logits.shape[0]
        target = torch.zeros(batch_size, self.num_classes - 1, device=logits.device)
        for i in range(batch_size):
            target[i, :labels[i]] = 1.0
        
        # Calculate binary cross entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
        
        return loss


class ViLa_MTL_MIL_Model(nn.Module):
    """
    Multi-Task Learning ViLa-MIL Model
    
    Main task: Cox regression (survival prediction) - no text prompts
    Auxiliary tasks: Classification/Grading tasks - with text prompts (only for prototype mode)
    
    All tasks share the same WSI-level image features.
    
    Modes:
    - prototype: Uses prototype-guided aggregation + text prompts (ViLa-MIL)
    - baseline: Uses pure MIL methods without text prompts (comparison baseline)
    """
    def __init__(self, config):
        super(ViLa_MTL_MIL_Model, self).__init__()
        self._actual_init(config)
    
    def get_trainable_state_dict(self):
        """
        Get only the trainable parameters. 
        - When Conch is frozen: exclude all text_encoder parameters (reduces model size from ~1.5GB to ~10MB)
        - When Conch is unfrozen: only save parameters with requires_grad=True
        """
        state_dict = {}
        
        # Check if using Conch and if it's unfrozen
        using_conch_unfrozen = False
        if hasattr(self, 'text_encoder') and hasattr(self.text_encoder, 'freeze_conch'):
            using_conch_unfrozen = not self.text_encoder.freeze_conch
        
        if using_conch_unfrozen:
            # Save only parameters that have requires_grad=True
            for name, param in self.named_parameters():
                if param.requires_grad:
                    state_dict[name] = param.data
        else:
            # 排除冻结的 CONCH/CLIP 主干以减小体积，但必须保留 text_encoder 下
            # 可训练的参数（ConchTextEncoder.projection）。否则 checkpoint 不含文本投影，
            # 推理时会退回随机初始化，导致文本注意力可视化每次结果都不同。
            trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
            for name, param in self.state_dict().items():
                if not name.startswith('text_encoder') or name in trainable_names:
                    state_dict[name] = param
        
        return state_dict
    
    def load_trainable_state_dict(self, state_dict):
        """
        Load only the trainable parameters from a saved state_dict
        """
        current_state = self.state_dict()
        for name, param in state_dict.items():
            if name in current_state:
                current_state[name].copy_(param)
        self.load_state_dict(current_state)

    def _actual_init(self, config):
        
        self.mil_aggregator_type = getattr(config, 'mil_aggregator', 'prototype')
        self.mil_aggregator = create_mil_aggregator(config)
        
        self.text_extractor = getattr(config, 'text_extractor', 'clip')
        self.use_text_prompts = getattr(config, 'use_text_prompts', True)
        
        # MDN configuration
        self.main_task_type = getattr(config, 'main_task', 'cox')
        
        # Check if task_configs exists and has length
        if hasattr(config, 'task_configs') and len(config.task_configs) > 0:
            self.task_configs = config.task_configs
            self.num_aux_tasks = len(config.task_configs)
        else:
            self.task_configs = []
            self.num_aux_tasks = 0
        
        # Zero-shot task configuration: {task_idx: [class_idx1, class_idx2, ...]}
        self.zero_shot_config = getattr(config, 'zero_shot_config', {})
        
        # Cross-scale contrastive learning setup
        self.use_cross_scale_contrastive = getattr(config, 'use_cross_scale_contrastive', False)
        if self.use_cross_scale_contrastive:
            from models.mil_aggregators import CrossScaleContrastiveLoss
            self.cross_scale_loss_fn = CrossScaleContrastiveLoss(
                temperature=getattr(config, 'contrast_temperature', 0.5),
                weight=getattr(config, 'cross_scale_weight', 0.1),
                intra_weight=getattr(config, 'contrast_intra_weight', 1.0),
                inter_weight=getattr(config, 'contrast_inter_weight', 0.5)
            )
            print(f"Model initialized with cross-scale contrastive learning enabled")
        
        # Consistency loss setup
        self.use_consistency = getattr(config, 'use_consistency', False)
        if self.use_consistency:
            from models.mil_aggregators import ConsistencyLoss
            self.consistency_loss_fn = ConsistencyLoss(
                noise_std=getattr(config, 'consistency_noise_std', 0.1)
            )
            self.consistency_weight = getattr(config, 'consistency_weight', 0.1)
            print(f"Model initialized with consistency loss enabled (weight={self.consistency_weight})")
        
        # Only initialize auxiliary task components if there are auxiliary tasks
        if self.num_aux_tasks > 0:
            if self.mil_aggregator_type == 'baseline':
                self._init_baseline_mode(config)
            else:
                self._init_prototype_mode(config)
        else:
            # No auxiliary tasks, initialize with None
            self.text_encoder = None
            self.clip_model = None
            self.prompt_learners = None
            self.cross_attention_text = None
            self.task_heads = None
            self.baseline_task_heads = None
        
        # Main task setup
        self.main_task_type = getattr(config, 'main_task', 'cox')
        if self.main_task_type == 'cox':
            self.cox_regressor = nn.ModuleDict({
                'low': nn.Linear(config.input_size, 1),
                'high': nn.Linear(config.input_size, 1)
            })
            self.cox_loss = CoxPHLoss()
        elif self.main_task_type == 'binary':
            self.binary_classifier = nn.ModuleDict({
                'low': nn.Linear(config.input_size, 2),
                'high': nn.Linear(config.input_size, 2)
            })
            self.binary_loss = nn.CrossEntropyLoss()
        elif self.main_task_type == 'mdn':
            # MDN for survival analysis
            mdn_K = getattr(config, 'mdn_K', 100)
            mdn_hidden_size = getattr(config, 'mdn_hidden_size', 256)
            mdn_param_share = getattr(config, 'mdn_param_share', ['mu', 'sigma'])
            mdn_dropout = getattr(config, 'mdn_dropout', 0.1)
            mdn_mode = getattr(config, 'mdn_mode', 'separate')
            mdn_mu_max = getattr(config, 'mdn_mu_max', 200.0)
            mdn_mu_min = getattr(config, 'mdn_mu_min', 0.0)
            mdn_sigma_init = getattr(config, 'mdn_sigma_init', 1.0)
            mdn_sigma_min = getattr(config, 'mdn_sigma_min', 4.0)
            # 新增：独立固定mu和sigma模式
            mdn_fixed_mu = getattr(config, 'mdn_fixed_mu', False)
            mdn_fixed_sigma = getattr(config, 'mdn_fixed_sigma', False)
            
            self.mdn_mode = mdn_mode
            self.mdn_fixed_mu = mdn_fixed_mu
            self.mdn_fixed_sigma = mdn_fixed_sigma
            
            if mdn_mode == 'separate':
                # Original: two independent MDNs
                self.mdn_regressor = nn.ModuleDict({
                    'low': SurvivalMDN(input_size=config.input_size, 
                                       hidden_size=mdn_hidden_size, 
                                       K=mdn_K, 
                                       param_share=mdn_param_share, 
                                       dropout=mdn_dropout,
                                       mu_min=mdn_mu_min,
                                       mu_max=mdn_mu_max,
                                       sigma_init=mdn_sigma_init,
                                       sigma_min=mdn_sigma_min,
                                       fixed_mu=mdn_fixed_mu,
                                       fixed_sigma=mdn_fixed_sigma),
                    'high': SurvivalMDN(input_size=config.input_size, 
                                        hidden_size=mdn_hidden_size, 
                                        K=mdn_K, 
                                        param_share=mdn_param_share, 
                                        dropout=mdn_dropout,
                                        mu_min=mdn_mu_min,
                                        mu_max=mdn_mu_max,
                                        sigma_init=mdn_sigma_init,
                                        sigma_min=mdn_sigma_min,
                                        fixed_mu=mdn_fixed_mu,
                                        fixed_sigma=mdn_fixed_sigma)
                })
            elif mdn_mode == 'unified':
                # Unified: fusion layer + single MDN
                self.mdn_fusion = nn.Sequential(
                    nn.Linear(2 * config.input_size, config.input_size),
                    nn.GELU(),
                    nn.Dropout(mdn_dropout)
                )
                self.mdn_regressor = SurvivalMDN(
                    input_size=config.input_size,
                    hidden_size=mdn_hidden_size,
                    K=mdn_K,
                    param_share=mdn_param_share,
                    dropout=mdn_dropout,
                    mu_min=mdn_mu_min,
                    mu_max=mdn_mu_max,
                    sigma_init=mdn_sigma_init,
                    sigma_min=mdn_sigma_min,
                    fixed_mu=mdn_fixed_mu,
                    fixed_sigma=mdn_fixed_sigma
                )
            elif mdn_mode == 'shared_weight':
                # Shared weight head + separate mu/sigma
                # Shared weight predictor
                self.mdn_weight_head = nn.Sequential(
                    nn.Linear(2 * config.input_size, mdn_hidden_size),
                    nn.GELU(),
                    nn.Dropout(mdn_dropout),
                    nn.Linear(mdn_hidden_size, mdn_K)
                )
                # Separate mu/sigma predictors
                self.mdn_mu_sigma_low = nn.Sequential(
                    nn.Linear(config.input_size, mdn_hidden_size),
                    nn.GELU(),
                    nn.Dropout(mdn_dropout),
                    nn.Linear(mdn_hidden_size, 2 * mdn_K)
                )
                self.mdn_mu_sigma_high = nn.Sequential(
                    nn.Linear(config.input_size, mdn_hidden_size),
                    nn.GELU(),
                    nn.Dropout(mdn_dropout),
                    nn.Linear(mdn_hidden_size, 2 * mdn_K)
                )
                # Fusion gate for mu/sigma
                self.mdn_fusion_gate = nn.Sequential(
                    nn.Linear(4 * mdn_K, mdn_K),
                    nn.Sigmoid()
                )
                # Shared mu/sigma parameters (similar to SurvivalMDN)
                self.mdn_shared_params = nn.ParameterDict()
                for param_name in mdn_param_share:
                    if param_name == 'mu':
                        self.mdn_shared_params[param_name] = nn.Parameter(
                            torch.linspace(mdn_mu_min, mdn_mu_max, mdn_K, dtype=torch.float32).unsqueeze(0))
                    elif param_name == 'sigma':
                        self.mdn_shared_params[param_name] = nn.Parameter(
                            torch.ones((1, mdn_K)) * inverse_softplus(torch.tensor(mdn_sigma_init)))
                self.mdn_K = mdn_K
                self.mdn_param_share = mdn_param_share
            else:
                raise ValueError(f"Unknown mdn_mode: {mdn_mode}")
            
            # Store MDN configuration
            self.mdn_mode = mdn_mode
            
            self.mdn_loss = MDNLoss()
            
            # Add method to get Gaussian parameters
            def get_gaussian_params(self):
                """
                Get the K Gaussian model parameters (mu and sigma) from the trained model.
                
                Returns:
                    dict with keys 'mu' and 'sigma', each being a numpy array of shape [K]
                """
                mu = None
                sigma = None
                
                if self.mdn_mode == 'separate':
                    # Separate mode: use the first MDN head (low resolution)
                    if hasattr(self, 'mdn_regressor') and self.mdn_regressor is not None:
                        if 'low' in self.mdn_regressor:
                            mdn_head = self.mdn_regressor['low']
                        else:
                            mdn_head = None
                            
                        if mdn_head is not None and hasattr(mdn_head, 'get_final_params'):
                            final_params = mdn_head.get_final_params()
                            if 'mu' in final_params:
                                mu = final_params['mu'].cpu().detach().numpy().squeeze()
                            if 'sigma' in final_params:
                                sigma = final_params['sigma'].cpu().detach().numpy().squeeze()
                
                elif self.mdn_mode == 'unified':
                    # Unified mode: use mdn_regressor
                    if hasattr(self, 'mdn_regressor') and self.mdn_regressor is not None:
                        mdn_regressor = self.mdn_regressor
                        if hasattr(mdn_regressor, 'get_final_params'):
                            final_params = mdn_regressor.get_final_params()
                            if 'mu' in final_params:
                                mu = final_params['mu'].cpu().detach().numpy().squeeze()
                            if 'sigma' in final_params:
                                sigma = final_params['sigma'].cpu().detach().numpy().squeeze()
                
                elif self.mdn_mode == 'shared_weight':
                    # Shared weight mode: mu and sigma are sample-specific, not shared
                    # Cannot extract shared mu/sigma for this mode
                    mu = None
                    sigma = None
                    print("Warning: shared_weight mode has sample-specific mu/sigma, cannot extract shared parameters")
                
                return {'mu': mu, 'sigma': sigma}
            
            # Monkey patch the method to the class
            ViLa_MTL_MIL_Model.get_gaussian_params = get_gaussian_params
        # else: main_task_type == 'none', no main task layers needed
        
        self.main_task_weight = getattr(config, 'main_task_weight', 1.0)
        self.aux_task_weights = getattr(config, 'aux_task_weights', None)
        if self.aux_task_weights is None:
            self.aux_task_weights = [1.0] * self.num_aux_tasks
        
        self.aux_task_loss_types = getattr(config, 'aux_task_losses', ['ce'] * self.num_aux_tasks)
        if self.num_aux_tasks > 0:
            self._init_loss_functions(config)
        else:
            self.aux_task_losses = []
    
    def _init_prototype_mode(self, config):
        """
        Initialize prototype mode: prototype-guided aggregation + text prompts (ViLa-MIL)
        When use_text_prompts=False, only initialize baseline task heads for traditional classification
        """
        if self.use_text_prompts:
            # Initialize text-related components only when using text prompts
            if self.text_extractor == 'clip':
                global clip
                if clip is None:
                    import clip
                clip_model, _ = clip.load("RN50", device="cpu")
                self.text_encoder = TextEncoder(clip_model.float())
                self.clip_model = clip_model.float()
            else:
                self.text_encoder = ConchTextEncoder(config)
                self.clip_model = None
            
            self.prompt_learners = nn.ModuleList()
            self.cross_attention_text = nn.ModuleList()
            self.task_heads = nn.ModuleList()
            self.baseline_task_heads = None
            
            for task_idx, task_config in enumerate(config.task_configs):
                if self.text_extractor == 'clip':
                    prompt_learner = PromptLearner(task_config['text_prompts'], self.clip_model, text_extractor='clip', config=config)
                else:
                    prompt_learner = PromptLearner(task_config['text_prompts'], text_extractor='conch', config=config)
                self.prompt_learners.append(prompt_learner)
                
                num_heads = getattr(config, 'num_heads', 1)
                cross_attn = MultiheadAttention(embed_dim=config.input_size, num_heads=num_heads)
                self.cross_attention_text.append(cross_attn)
                
                num_classes = task_config['num_classes']
                task_head = nn.ModuleDict({
                    'low': nn.Linear(config.input_size, num_classes),
                    'high': nn.Linear(config.input_size, num_classes)
                })
                self.task_heads.append(task_head)
        else:
            # When not using text prompts, only initialize baseline task heads
            self.text_encoder = None
            self.clip_model = None
            self.prompt_learners = None
            self.cross_attention_text = None
            self.task_heads = None
            self.baseline_task_heads = nn.ModuleList()
            
            for task_idx, task_config in enumerate(config.task_configs):
                num_classes = task_config['num_classes']
                baseline_task_head = nn.ModuleDict({
                    'low': nn.Linear(config.input_size, num_classes),
                    'high': nn.Linear(config.input_size, num_classes)
                })
                self.baseline_task_heads.append(baseline_task_head)
    
    def _init_baseline_mode(self, config):
        """
        Initialize baseline mode: pure MIL methods without text prompts (comparison baseline)
        """
        self.text_encoder = None
        self.clip_model = None
        self.prompt_learners = None
        self.cross_attention_text = None
        self.task_heads = None
        
        self.baseline_task_heads = nn.ModuleList()
        for task_idx, task_config in enumerate(config.task_configs):
            num_classes = task_config['num_classes']
            task_head = nn.ModuleDict({
                'low': nn.Linear(config.input_size, num_classes),
                'high': nn.Linear(config.input_size, num_classes)
            })
            self.baseline_task_heads.append(task_head)
    
    def _init_loss_functions(self, config):
        """
        Initialize loss functions for all auxiliary tasks
        """
        self.aux_task_losses = []
        for task_idx, task_config in enumerate(config.task_configs):
            loss_type = self.aux_task_loss_types[task_idx]
            num_classes = task_config['num_classes']
            
            if loss_type == 'ce':
                loss_fn = nn.CrossEntropyLoss()
            elif loss_type == 'ordinal':
                loss_fn = OrdinalLoss(num_classes)
            elif loss_type == 'mse':
                loss_fn = nn.MSELoss()
            elif loss_type == 'focal':
                class FocalLoss(nn.Module):
                    def __init__(self, alpha=1, gamma=2, reduction='mean'):
                        super(FocalLoss, self).__init__()
                        self.alpha = alpha
                        self.gamma = gamma
                        self.reduction = reduction
                    
                    def forward(self, inputs, targets):
                        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                        pt = torch.exp(-ce_loss)
                        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                        
                        if self.reduction == 'mean':
                            return focal_loss.mean()
                        elif self.reduction == 'sum':
                            return focal_loss.sum()
                        else:
                            return focal_loss
                
                loss_fn = FocalLoss()
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            self.aux_task_losses.append(loss_fn)
    
    def extract_shared_image_features(self, x_s, x_l):
        """
        Extract shared WSI-level image features from low and high resolution patches.
        This is the shared backbone for all tasks.
        """
        # Use MIL aggregator to process features
        return self.mil_aggregator(x_s, x_l)
    
    def forward(self, x_s_list, coord_s_list, x_l_list, coords_l_list, 
                cox_durations=None, cox_events=None,
                main_task_labels=None,
                aux_labels=None):
        """
        Forward pass for multi-task learning with batch processing.
        
        Args:
            x_s_list: List of low resolution patch features for each sample in batch
            coord_s_list: List of low resolution patch coordinates for each sample in batch
            x_l_list: List of high resolution patch features for each sample in batch
            coords_l_list: List of high resolution patch coordinates for each sample in batch
            cox_durations: Survival durations for Cox regression (main task) [batch_size]
            cox_events: Event indicators for Cox regression (main task) [batch_size]
            main_task_labels: Labels for binary main task [batch_size]
            aux_labels: List of labels for auxiliary tasks [task1_labels, task2_labels, ...]
        
        Returns:
            Dictionary containing predictions and losses for all tasks
        """
        # Get device from model parameters to ensure consistency
        device = next(self.parameters()).device if hasattr(self, 'cox_regressor') or hasattr(self, 'binary_classifier') else torch.device('cpu')
        batch_size = len(x_s_list)
        
        # Move all input tensors to the correct device
        x_s_list = [x_s.to(device) for x_s in x_s_list]
        coord_s_list = [coord_s.to(device) for coord_s in coord_s_list]
        x_l_list = [x_l.to(device) for x_l in x_l_list]
        coords_l_list = [coords_l.to(device) for coords_l in coords_l_list]
        
        if cox_durations is not None:
            cox_durations = cox_durations.to(device)
        if cox_events is not None:
            cox_events = cox_events.to(device)
        if main_task_labels is not None:
            main_task_labels = main_task_labels.to(device)
        
        # Move aux_labels to the correct device if provided
        if aux_labels is not None:
            aux_labels = [labels.to(device) if labels is not None else None for labels in aux_labels]
        
        # Process each sample in the batch
        batch_image_features_low = []
        batch_image_features_high = []
        batch_compents_low = []
        batch_compents_high = []
        batch_M_low = []
        batch_M_high = []
        
        # Extract features for all samples in batch
        for i in range(batch_size):
            x_s = x_s_list[i]
            coord_s = coord_s_list[i]
            x_l = x_l_list[i]
            coords_l = coords_l_list[i]
            
            # Extract shared image features
            image_features_low, image_features_high, compents_low, compents_high, M_low, M_high = \
                self.extract_shared_image_features(x_s, x_l)
            
            batch_image_features_low.append(image_features_low)
            batch_image_features_high.append(image_features_high)
            batch_compents_low.append(compents_low)
            batch_compents_high.append(compents_high)
            batch_M_low.append(M_low)
            batch_M_high.append(M_high)
        
        # Stack features for batch processing
        batch_image_features_low = torch.stack(batch_image_features_low)
        batch_image_features_high = torch.stack(batch_image_features_high)
        
        # Ensure features have correct shape for batch processing
        # Shape should be [batch_size, feature_dim]
        if batch_image_features_low.dim() == 3:
            batch_image_features_low = batch_image_features_low.squeeze(1)
        if batch_image_features_high.dim() == 3:
            batch_image_features_high = batch_image_features_high.squeeze(1)

        # Compute cross-scale contrastive loss (only during training)
        contrastive_loss = torch.tensor(0.0, device=batch_image_features_low.device)
        if self.use_cross_scale_contrastive and self.training:
            #print(f"Model-level contrastive loss computation - batch_size: {batch_size}, features shape: {batch_image_features_low.shape}")
            contrastive_loss = self.cross_scale_loss_fn(batch_image_features_low, batch_image_features_high)
            #print(f"Contrastive loss: {contrastive_loss.item()}")

        # Initialize outputs
        outputs = {
            'main_task': {},
            'aux_tasks': [],
            'contrastive_loss': contrastive_loss
        }
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Main task processing
        if self.main_task_type == 'cox':
            # Cox regression task
            if cox_durations is not None and cox_events is not None:
                # Use both low and high resolution features
                risk_low = self.cox_regressor['low'](batch_image_features_low).squeeze()
                risk_high = self.cox_regressor['high'](batch_image_features_high).squeeze()
                risk_scores = (risk_low + risk_high) / 2
                
                # Compute loss for low, high, and combined
                cox_loss_low = self.cox_loss(risk_low, cox_durations, cox_events)
                cox_loss_high = self.cox_loss(risk_high, cox_durations, cox_events)
                cox_loss = self.cox_loss(risk_scores, cox_durations, cox_events)
                total_loss = total_loss + self.main_task_weight * cox_loss
                
                outputs['main_task'] = {
                    'risk_scores': risk_scores,
                    'risk_scores_low': risk_low,
                    'risk_scores_high': risk_high,
                    'loss': cox_loss,
                    'loss_low': cox_loss_low,
                    'loss_high': cox_loss_high
                }
        elif self.main_task_type == 'binary':
            # Binary classification task
            if main_task_labels is not None:
                # Use both low and high resolution features
                logits_low = self.binary_classifier['low'](batch_image_features_low)
                logits_high = self.binary_classifier['high'](batch_image_features_high)
                logits = logits_low + logits_high
                
                # Compute loss for low, high, and combined
                binary_loss_low = self.binary_loss(logits_low, main_task_labels)
                binary_loss_high = self.binary_loss(logits_high, main_task_labels)
                binary_loss = self.binary_loss(logits, main_task_labels)
                total_loss = total_loss + self.main_task_weight * binary_loss
                
                # Compute predictions and probabilities
                Y_prob = F.softmax(logits, dim=1)
                Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze()
                
                outputs['main_task'] = {
                    'logits': logits,
                    'logits_low': logits_low,
                    'logits_high': logits_high,
                    'prob': Y_prob,
                    'pred': Y_hat,
                    'labels': main_task_labels,
                    'loss': binary_loss,
                    'loss_low': binary_loss_low,
                    'loss_high': binary_loss_high
                }
        elif self.main_task_type == 'mdn':
            # MDN survival analysis task
            if cox_durations is not None and cox_events is not None:
                # Create constant_dict for MDN
                device = cox_durations.device
                
                eval_t = torch.linspace(0.1, 100.0, 1000).to(device)
                t_min = torch.tensor(0.1, device=device)
                t_max = torch.tensor(100.0, device=device)
                
                constant_dict = {
                    "eval_t": eval_t,
                    "t_min": t_min,
                    "t_max": t_max,
                    "t_max_0.1": torch.tensor(90.0, device=device),
                    "t_max_0.2": torch.tensor(80.0, device=device),
                    "t_max_0.3": torch.tensor(70.0, device=device),
                    "t_max_0.4": torch.tensor(60.0, device=device),
                    "t_max_0.5": torch.tensor(50.0, device=device),
                    "NUM_INT_STEPS": torch.tensor(1000, device=device)
                }
                
                # Convert cox_events to censorship indicator for MDN loss
                # cox_events: 1 = event occurred (uncensored), 0 = censored  
                # censorship (c): 0 = uncensored, 1 = censored
                # Loss formula: -(1-c)*log(pdf) - log(survival_func)
                # When c=0 (uncensored): loss = -log(pdf)
                # When c=1 (censored): loss = -log(survival_func)
                censorship = 1.0 - cox_events
                
                if self.mdn_mode == 'separate':
                    # Original: two independent MDNs
                    # Use both low and high resolution features
                    input_low = {'x': batch_image_features_low}
                    input_high = {'x': batch_image_features_high}
                    
                    # Use train_step method
                    train_outputs_low, eval_outputs_low = self.mdn_regressor['low'].train_step(input_low, cox_durations, censorship, constant_dict)
                    train_outputs_high, eval_outputs_high = self.mdn_regressor['high'].train_step(input_high, cox_durations, censorship, constant_dict)
                    
                    # Compute loss
                    mdn_loss_low = self.mdn_loss(train_outputs_low)
                    mdn_loss_high = self.mdn_loss(train_outputs_high)
                    mdn_loss = (mdn_loss_low + mdn_loss_high) / 2
                    total_loss = total_loss + self.main_task_weight * mdn_loss
                    
                    # Combine eval outputs for evaluation
                    eval_outputs = {}
                    for key in eval_outputs_low:
                        if key in ['t', 'c']:
                            eval_outputs[key] = eval_outputs_low[key]
                        elif key in ['eval_t', 'cum_hazard_seqs', 'survival_seqs'] or 'survival_seqs_' in key:
                            # Average outputs from low and high
                            eval_outputs[key] = (eval_outputs_low[key] + eval_outputs_high[key]) / 2
                    
                    outputs['main_task'] = {
                        'params': train_outputs_low,  # Use low params for prediction
                        'params_low': train_outputs_low,
                        'params_high': train_outputs_high,
                        'eval_outputs': eval_outputs,
                        'loss': mdn_loss,
                        'loss_low': mdn_loss_low,
                        'loss_high': mdn_loss_high,
                        'durations': cox_durations,
                        'events': cox_events
                    }
                    
                elif self.mdn_mode == 'unified':
                    # Unified: fusion layer + single MDN
                    # Fuse low and high resolution features
                    fused_features = torch.cat([batch_image_features_low, batch_image_features_high], dim=1)
                    fused_features = self.mdn_fusion(fused_features)
                    
                    # Single MDN prediction
                    input_dict = {'x': fused_features}
                    train_outputs, eval_outputs = self.mdn_regressor.train_step(input_dict, cox_durations, censorship, constant_dict)
                    
                    # Get raw MDN params (w, mu, sigma) for saving
                    mdn_params, _ = self.mdn_regressor(**input_dict)
                    
                    # Compute loss
                    mdn_loss = self.mdn_loss(train_outputs)
                    total_loss = total_loss + self.main_task_weight * mdn_loss
                    
                    outputs['main_task'] = {
                        'params': mdn_params,  # Use raw params for saving weights
                        'train_outputs': train_outputs,  # Keep train outputs for loss
                        'eval_outputs': eval_outputs,
                        'loss': mdn_loss,
                        'loss_low': mdn_loss,  # For compatibility
                        'loss_high': mdn_loss,  # For compatibility
                        'durations': cox_durations,
                        'events': cox_events
                    }
                    
                elif self.mdn_mode == 'shared_weight':
                    # Shared weight head + separate mu/sigma
                    # Predict unified weights from fused features
                    fused_features = torch.cat([batch_image_features_low, batch_image_features_high], dim=1)
                    weights_logits = self.mdn_weight_head(fused_features)
                    weights = weights_logits.softmax(-1)  # [batch, K]
                    
                    # Predict separate mu/sigma for low and high
                    mu_sigma_low = self.mdn_mu_sigma_low(batch_image_features_low)
                    mu_sigma_high = self.mdn_mu_sigma_high(batch_image_features_high)
                    
                    mu_low, sigma_low = mu_sigma_low.chunk(2, dim=1)
                    mu_high, sigma_high = mu_sigma_high.chunk(2, dim=1)
                    
                    # Apply softplus and clamp to sigma
                    sigma_low = F.softplus(sigma_low).clamp(min=1e-2, max=1e2)
                    sigma_high = F.softplus(sigma_high).clamp(min=1e-2, max=1e2)
                    
                    # Clamp mu
                    mu_low = mu_low.clamp(min=-200, max=200)
                    mu_high = mu_high.clamp(min=-200, max=200)
                    
                    # Fusion gate for mu/sigma
                    fusion_input = torch.cat([mu_low, sigma_low, mu_high, sigma_high], dim=1)
                    gate = self.mdn_fusion_gate(fusion_input)  # [batch, K]
                    
                    # Weighted fusion of mu and sigma
                    mu = gate * mu_low + (1 - gate) * mu_high
                    sigma = gate * sigma_low + (1 - gate) * sigma_high
                    
                    # Build params dict
                    params = {
                        'w': weights,
                        'mu': mu,
                        'sigma': sigma
                    }
                    
                    # Compute CDF and survival function for loss
                    survival_func = 1 - self._mdn_cdf(params, cox_durations)
                    pdf = torch.exp(self._mdn_log_prob(params, cox_durations))
                    
                    train_outputs = {
                        't': cox_durations,
                        'c': censorship,
                        'pdf': pdf,
                        'survival_func': survival_func,
                        'subloss': None
                    }

                    # Compute loss
                    mdn_loss = self.mdn_loss(train_outputs)
                    total_loss = total_loss + self.main_task_weight * mdn_loss

                    # Compute eval outputs
                    with torch.no_grad():
                        eval_outputs = self._mdn_eval_step(params, cox_durations, censorship, constant_dict)
                    
                    outputs['main_task'] = {
                        'params': train_outputs,
                        'eval_outputs': eval_outputs,
                        'loss': mdn_loss,
                        'loss_low': mdn_loss,  # For compatibility
                        'loss_high': mdn_loss,  # For compatibility
                        'durations': cox_durations,
                        'events': cox_events
                    }
        # else: main_task_type == 'none', no main task processing
        
        # Auxiliary tasks: Classification tasks
        # Different processing based on mode (prototype vs baseline)
        if self.num_aux_tasks > 0:
            if self.mil_aggregator_type == 'baseline':
                # Baseline mode: pure MIL without text prompts
                for task_idx in range(self.num_aux_tasks):
                    task_output = {}
                    task_config = self.task_configs[task_idx]
                    
                    # Use baseline task heads (direct classification)
                    logits_low = self.baseline_task_heads[task_idx]['low'](batch_image_features_low)
                    logits_high = self.baseline_task_heads[task_idx]['high'](batch_image_features_high)
                    logits = logits_low + logits_high
                    
                    task_output['logits'] = logits
                    task_output['logits_low'] = logits_low
                    task_output['logits_high'] = logits_high
                    
                    # Check if this task has zero-shot classes
                    zero_shot_classes = self.zero_shot_config.get(task_idx, [])
                    
                    # Compute loss if labels are provided
                    if aux_labels is not None and aux_labels[task_idx] is not None:
                        labels = aux_labels[task_idx]
                        
                        if labels.dim() > 1:
                            labels = labels.squeeze()
                        
                        if labels.dtype != torch.long:
                            labels = labels.long()
                        
                        labels = labels.to(logits.device)
                        
                        # For zero-shot tasks, only compute loss for non-zero-shot classes
                        if zero_shot_classes:
                            # Create mask for non-zero-shot samples
                            non_zero_shot_mask = ~torch.isin(labels, torch.tensor(zero_shot_classes, device=labels.device))
                            
                            if non_zero_shot_mask.any():
                                # Only compute loss for non-zero-shot samples
                                non_zero_shot_logits = logits[non_zero_shot_mask]
                                non_zero_shot_logits_low = logits_low[non_zero_shot_mask]
                                non_zero_shot_logits_high = logits_high[non_zero_shot_mask]
                                non_zero_shot_labels = labels[non_zero_shot_mask]
                                
                                loss_fn = self.aux_task_losses[task_idx]
                                task_loss = loss_fn(non_zero_shot_logits, non_zero_shot_labels)
                                task_loss_low = loss_fn(non_zero_shot_logits_low, non_zero_shot_labels)
                                task_loss_high = loss_fn(non_zero_shot_logits_high, non_zero_shot_labels)
                                total_loss = total_loss + self.aux_task_weights[task_idx] * task_loss
                                
                                task_output['loss'] = task_loss
                                task_output['loss_low'] = task_loss_low
                                task_output['loss_high'] = task_loss_high
                            else:
                                # All samples are zero-shot, no loss to compute
                                task_output['loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                                task_output['loss_low'] = torch.tensor(0.0, device=device, requires_grad=True)
                                task_output['loss_high'] = torch.tensor(0.0, device=device, requires_grad=True)
                        else:
                            # No zero-shot classes, compute loss normally
                            loss_fn = self.aux_task_losses[task_idx]
                            task_loss = loss_fn(logits, labels)
                            task_loss_low = loss_fn(logits_low, labels)
                            task_loss_high = loss_fn(logits_high, labels)
                            total_loss = total_loss + self.aux_task_weights[task_idx] * task_loss
                            
                            task_output['loss'] = task_loss
                            task_output['loss_low'] = task_loss_low
                            task_output['loss_high'] = task_loss_high
                        
                        Y_prob = F.softmax(logits, dim=1)
                        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze()
                        
                        task_output['prob'] = Y_prob
                        task_output['pred'] = Y_hat
                        task_output['labels'] = labels
                        task_output['zero_shot_mask'] = torch.isin(labels, torch.tensor(zero_shot_classes, device=labels.device)) if zero_shot_classes else None
                    else:
                        # No labels provided (e.g., during inference), just compute predictions
                        Y_prob = F.softmax(logits, dim=1)
                        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze()
                        
                        task_output['prob'] = Y_prob
                        task_output['pred'] = Y_hat
                    
                    outputs['aux_tasks'].append(task_output)
            else:
                # Prototype mode: either text-guided classification or traditional classification
                for task_idx in range(self.num_aux_tasks):
                    task_output = {}
                    
                    if self.use_text_prompts:
                        # Text-guided classification (ViLa-MIL)
                        # Get text prompts for this task
                        prompts = self.prompt_learners[task_idx]()
                        
                        # Get text features based on text extractor type
                        if self.text_extractor == 'clip':
                            tokenized_prompts = self.prompt_learners[task_idx].tokenized_prompts
                            text_features = self.text_encoder(prompts, tokenized_prompts)
                        else:  # conch
                            text_features = self.text_encoder(prompts)
                        
                        task_config = self.task_configs[task_idx]
                        num_classes = task_config['num_classes']
                        
                        # 每个类有两个描述（低分辨率和高分辨率），所以text_features的长度是num_classes * 2
                        # text_features_low使用每个类的低分辨率描述
                        text_features_low = text_features[:num_classes]
                        # text_features_high使用每个类的高分辨率描述
                        text_features_high = text_features[num_classes:num_classes*2]
                        
                        batch_logits = []
                        batch_logits_low = []
                        batch_logits_high = []
                        batch_labels = []
                        
                        for i in range(batch_size):
                            image_features_low = batch_image_features_low[i]
                            image_features_high = batch_image_features_high[i]
                            compents_low = batch_compents_low[i]
                            compents_high = batch_compents_high[i]
                            M_low = batch_M_low[i]
                            M_high = batch_M_high[i]
                            
                            # Use compents if available (prototype mode), otherwise use raw patch features (AMIL mode)
                            if compents_low.size(0) > 0:
                                image_context_low = torch.cat((compents_low.squeeze(), M_low), dim=0)
                            else:
                                image_context_low = M_low
                            text_context_features_low, _ = self.cross_attention_text[task_idx](
                                text_features_low.unsqueeze(1), image_context_low, image_context_low
                            )
                            text_features_low_enhanced = text_context_features_low.squeeze() + text_features_low
                            
                            # Use compents if available (prototype mode), otherwise use raw patch features (AMIL mode)
                            if compents_high.size(0) > 0:
                                image_context_high = torch.cat((compents_high.squeeze(), M_high), dim=0)
                            else:
                                image_context_high = M_high
                            text_context_features_high, _ = self.cross_attention_text[task_idx](
                                text_features_high.unsqueeze(1), image_context_high, image_context_high
                            )
                            text_features_high_enhanced = text_context_features_high.squeeze() + text_features_high
                            
                            img_feat_low = image_features_low.view(1, -1)
                            img_feat_high = image_features_high.view(1, -1)
                            
                            logits_low = img_feat_low @ text_features_low_enhanced.T
                            logits_high = img_feat_high @ text_features_high_enhanced.T
                            logits = logits_low + logits_high
                            
                            batch_logits.append(logits)
                            batch_logits_low.append(logits_low)
                            batch_logits_high.append(logits_high)
                            
                            if aux_labels is not None and aux_labels[task_idx] is not None:
                                label = aux_labels[task_idx][i].unsqueeze(0)
                                batch_labels.append(label)
                        
                        batch_logits = torch.stack(batch_logits)
                        batch_logits_low = torch.stack(batch_logits_low)
                        batch_logits_high = torch.stack(batch_logits_high)
                        
                        if batch_logits.dim() == 3:
                            batch_logits = batch_logits.squeeze(1)
                            batch_logits_low = batch_logits_low.squeeze(1)
                            batch_logits_high = batch_logits_high.squeeze(1)
                        elif batch_logits.dim() == 1:
                            batch_logits = batch_logits.unsqueeze(1)
                            batch_logits_low = batch_logits_low.unsqueeze(1)
                            batch_logits_high = batch_logits_high.unsqueeze(1)
                        
                        task_output['logits'] = batch_logits
                        task_output['logits_low'] = batch_logits_low
                        task_output['logits_high'] = batch_logits_high
                        
                        # Check if this task has zero-shot classes
                        zero_shot_classes = self.zero_shot_config.get(task_idx, [])
                        
                        if aux_labels is not None and aux_labels[task_idx] is not None:
                            labels = aux_labels[task_idx]
                            
                            if labels.dim() > 1:
                                labels = labels.squeeze()
                            
                            if labels.dtype != torch.long:
                                labels = labels.long()
                            
                            labels = labels.to(batch_logits.device)
                            
                            # For zero-shot tasks, only compute loss for non-zero-shot classes
                            if zero_shot_classes:
                                # Create mask for non-zero-shot samples
                                non_zero_shot_mask = ~torch.isin(labels, torch.tensor(zero_shot_classes, device=labels.device))
                                
                                if non_zero_shot_mask.any():
                                    # Only compute loss for non-zero-shot samples
                                    non_zero_shot_logits = batch_logits[non_zero_shot_mask]
                                    non_zero_shot_logits_low = batch_logits_low[non_zero_shot_mask]
                                    non_zero_shot_logits_high = batch_logits_high[non_zero_shot_mask]
                                    non_zero_shot_labels = labels[non_zero_shot_mask]
                                    
                                    loss_fn = self.aux_task_losses[task_idx]
                                    task_loss = loss_fn(non_zero_shot_logits, non_zero_shot_labels)
                                    task_loss_low = loss_fn(non_zero_shot_logits_low, non_zero_shot_labels)
                                    task_loss_high = loss_fn(non_zero_shot_logits_high, non_zero_shot_labels)
                                    total_loss = total_loss + self.aux_task_weights[task_idx] * task_loss
                                    
                                    task_output['loss'] = task_loss
                                    task_output['loss_low'] = task_loss_low
                                    task_output['loss_high'] = task_loss_high
                                else:
                                    # All samples are zero-shot, no loss to compute
                                    task_output['loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                                    task_output['loss_low'] = torch.tensor(0.0, device=device, requires_grad=True)
                                    task_output['loss_high'] = torch.tensor(0.0, device=device, requires_grad=True)
                            else:
                                # No zero-shot classes, compute loss normally
                                loss_fn = self.aux_task_losses[task_idx]
                                task_loss = loss_fn(batch_logits, labels)
                                task_loss_low = loss_fn(batch_logits_low, labels)
                                task_loss_high = loss_fn(batch_logits_high, labels)
                                total_loss = total_loss + self.aux_task_weights[task_idx] * task_loss
                                
                                task_output['loss'] = task_loss
                                task_output['loss_low'] = task_loss_low
                                task_output['loss_high'] = task_loss_high
                            
                            Y_prob = F.softmax(batch_logits, dim=1)
                            Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze()
                            
                            task_output['prob'] = Y_prob
                            task_output['pred'] = Y_hat
                            task_output['labels'] = labels
                            task_output['zero_shot_mask'] = torch.isin(labels, torch.tensor(zero_shot_classes, device=labels.device)) if zero_shot_classes else None
                        else:
                            # No labels provided (e.g., during inference), just compute predictions
                            Y_prob = F.softmax(batch_logits, dim=1)
                        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze()
                        
                        task_output['prob'] = Y_prob
                        task_output['pred'] = Y_hat
                    else:
                        # Traditional classification without text prompts
                        task_config = self.task_configs[task_idx]
                        
                        # Use baseline task heads (direct classification)
                        logits_low = self.baseline_task_heads[task_idx]['low'](batch_image_features_low)
                        logits_high = self.baseline_task_heads[task_idx]['high'](batch_image_features_high)
                        logits = logits_low + logits_high
                        
                        task_output['logits'] = logits
                        task_output['logits_low'] = logits_low
                        task_output['logits_high'] = logits_high
                        
                        # Check if this task has zero-shot classes
                        zero_shot_classes = self.zero_shot_config.get(task_idx, [])
                        
                        # Compute loss if labels are provided
                        if aux_labels is not None and aux_labels[task_idx] is not None:
                            labels = aux_labels[task_idx]
                            
                            if labels.dim() > 1:
                                labels = labels.squeeze()
                            
                            if labels.dtype != torch.long:
                                labels = labels.long()
                            
                            labels = labels.to(logits.device)
                            
                            # For zero-shot tasks, only compute loss for non-zero-shot classes
                            if zero_shot_classes:
                                # Create mask for non-zero-shot samples
                                non_zero_shot_mask = ~torch.isin(labels, torch.tensor(zero_shot_classes, device=labels.device))
                                
                                if non_zero_shot_mask.any():
                                    # Only compute loss for non-zero-shot samples
                                    non_zero_shot_logits = logits[non_zero_shot_mask]
                                    non_zero_shot_logits_low = logits_low[non_zero_shot_mask]
                                    non_zero_shot_logits_high = logits_high[non_zero_shot_mask]
                                    non_zero_shot_labels = labels[non_zero_shot_mask]
                                    
                                    loss_fn = self.aux_task_losses[task_idx]
                                    task_loss = loss_fn(non_zero_shot_logits, non_zero_shot_labels)
                                    task_loss_low = loss_fn(non_zero_shot_logits_low, non_zero_shot_labels)
                                    task_loss_high = loss_fn(non_zero_shot_logits_high, non_zero_shot_labels)
                                    total_loss = total_loss + self.aux_task_weights[task_idx] * task_loss
                                    
                                    task_output['loss'] = task_loss
                                    task_output['loss_low'] = task_loss_low
                                    task_output['loss_high'] = task_loss_high
                                else:
                                    # All samples are zero-shot, no loss to compute
                                    task_output['loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                                    task_output['loss_low'] = torch.tensor(0.0, device=device, requires_grad=True)
                                    task_output['loss_high'] = torch.tensor(0.0, device=device, requires_grad=True)
                            else:
                                # No zero-shot classes, compute loss normally
                                loss_fn = self.aux_task_losses[task_idx]
                                task_loss = loss_fn(logits, labels)
                                task_loss_low = loss_fn(logits_low, labels)
                                task_loss_high = loss_fn(logits_high, labels)
                                total_loss = total_loss + self.aux_task_weights[task_idx] * task_loss
                                
                                task_output['loss'] = task_loss
                                task_output['loss_low'] = task_loss_low
                                task_output['loss_high'] = task_loss_high
                            
                            Y_prob = F.softmax(logits, dim=1)
                            Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze()
                            
                            task_output['prob'] = Y_prob
                            task_output['pred'] = Y_hat
                            task_output['labels'] = labels
                            task_output['zero_shot_mask'] = torch.isin(labels, torch.tensor(zero_shot_classes, device=labels.device)) if zero_shot_classes else None
                        else:
                            # No labels provided (e.g., during inference), just compute predictions
                            Y_prob = F.softmax(logits, dim=1)
                            Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze()
                            
                            task_output['prob'] = Y_prob
                            task_output['pred'] = Y_hat
                    
                    outputs['aux_tasks'].append(task_output)
        
        # Add cross-scale contrastive loss if enabled
        if self.use_cross_scale_contrastive and self.training:
            total_loss = total_loss + outputs['contrastive_loss']
        
        # Add consistency loss if enabled
        if self.use_consistency and self.training:
            consistency_loss = self.consistency_loss_fn(batch_image_features_low) + \
                             self.consistency_loss_fn(batch_image_features_high)
            consistency_loss = consistency_loss / 2
            outputs['consistency_loss'] = consistency_loss
            total_loss = total_loss + self.consistency_weight * consistency_loss
        
        outputs['total_loss'] = total_loss
        
        return outputs
    
    def _mdn_cdf(self, params, times):
        """Compute CDF for MDN params (used in shared_weight mode)"""
        from models.mdn import safe_inverse_softplus
        ws, mus, sigmas = params['w'], params['mu'], params['sigma']
        device = ws.device
        
        times = times.to(device)
        inv_softplus_times = safe_inverse_softplus(times)
        inv_softplus_times = inv_softplus_times.unsqueeze(-1)
        
        K = ws.shape[1]
        if len(times.shape) == 1:
            normal_dists = torch.distributions.Normal(mus, sigmas)
            repeated_inv_softplus_times = inv_softplus_times.repeat(1, K)
            log_normal_cdfs = normal_dists.cdf(repeated_inv_softplus_times)
            cdfs = ws * log_normal_cdfs
            cdfs = cdfs.sum(-1)
        else:
            time_len = times.shape[-1]
            mus_expanded = mus.unsqueeze(1).repeat(1, time_len, 1)
            sigmas_expanded = sigmas.unsqueeze(1).repeat(1, time_len, 1)
            normal_dists = torch.distributions.Normal(mus_expanded, sigmas_expanded)
            repeated_inv_softplus_times = inv_softplus_times.repeat(1, 1, K)
            log_normal_cdfs = normal_dists.cdf(repeated_inv_softplus_times)
            ws_expanded = ws.unsqueeze(1)
            cdfs = ws_expanded * log_normal_cdfs
            cdfs = cdfs.sum(-1)
            cdfs[cdfs > 1.] = 1.
        return cdfs
    
    def _mdn_log_prob(self, params, times):
        """Compute log probability for MDN params (used in shared_weight mode)"""
        from models.mdn import safe_inverse_softplus, logsumexp, inverse_softplus_grad
        ws, mus, sigmas = params['w'], params['mu'], params['sigma']
        device = ws.device
        
        times = times.to(device)
        inv_softplus_times = safe_inverse_softplus(times)
        inv_softplus_times = inv_softplus_times.unsqueeze(-1)
        K = ws.shape[1]
        normal_dists = torch.distributions.Normal(mus, sigmas)
        repeated_inv_softplus_times = inv_softplus_times.repeat(1, 1, K)
        log_normal_pdfs = normal_dists.log_prob(repeated_inv_softplus_times)
        log_pdfs = logsumexp(a=log_normal_pdfs, dim=-1, b=ws).squeeze()
        log_grad = torch.log(inverse_softplus_grad(times))
        return log_pdfs + log_grad
    
    def _mdn_eval_step(self, params, t, c, constant_dict):
        """Eval step for shared_weight mode"""
        with torch.no_grad():
            outputs = {}
            outputs["t"] = t
            outputs['c'] = c
            outputs["eval_t"] = constant_dict["eval_t"]
            batch_size = 1
            device = t.device
            
            # Eval for time-dependent C-index
            eval_t = constant_dict["eval_t"].repeat(batch_size, 1).to(device)
            cdf = self._mdn_cdf(params, eval_t)
            survival_func = 1 - cdf
            cum_hazard = - torch.log(survival_func)
            outputs["cum_hazard_seqs"] = cum_hazard.transpose(1, 0)
            
            # Eval for Brier Score
            t_min = constant_dict["t_min"]
            t_max = constant_dict["t_max"]
            t_seq = torch.linspace(
                t_min.item(), t_max.item(), constant_dict['NUM_INT_STEPS'].item(), dtype=t_min.dtype,
                device=device)
            eval_t = t_seq.unsqueeze(0).repeat(batch_size, 1)
            cdf = self._mdn_cdf(params, eval_t)
            survival_func = 1 - cdf
            outputs["survival_seqs"] = survival_func.transpose(1, 0)
            
            for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
                t_min = constant_dict["t_min"]
                t_max = constant_dict["t_max_{}".format(eps)]
                t_seq = torch.linspace(
                    t_min.item(), t_max.item(), constant_dict['NUM_INT_STEPS'].item(), dtype=t_min.dtype,
                    device=device)
                eval_t = t_seq.unsqueeze(0).repeat(batch_size, 1)
                cdf = self._mdn_cdf(params, eval_t)
                survival_func = 1 - cdf
                outputs["survival_seqs_{}".format(eps)] = survival_func.transpose(1, 0)
        return outputs