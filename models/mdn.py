import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)

def safe_inverse_softplus(x):
    return torch.log(torch.exp(x) - 1 + 1e-6)

def inverse_softplus_grad(x):
    return torch.exp(x) / (torch.exp(x) - 1 + 1e-6)

def logsumexp(a, dim, b):
    a_max = torch.max(a, dim=dim, keepdims=True)[0]
    out = torch.log(torch.sum(b * torch.exp(a - a_max), dim=dim, keepdims=True) + 1e-6)
    out += a_max
    return out

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers=2, dropout_rate=0.25):
        super(MLP, self).__init__()
        self.hidden_layers = hidden_layers
        layers = []
        for i in range(self.hidden_layers):
            if i == 0:
                in_size = input_size
            else:
                in_size = hidden_size
            layers.extend([nn.Linear(in_size, hidden_size), nn.GELU(), nn.Dropout(dropout_rate)])
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.layers(x)
        o = self.fc(h)
        return o


class SurvivalMDN(nn.Module):
    """
    Mixture Density Network for Survival Analysis
    Adapted from SCMIL project for integration with ViLa-MIL
    
    支持独立固定mu和sigma的模式：
    - fixed_mu=True: mu固定为指定范围内的均匀分布，不参与训练
    - fixed_sigma=True: sigma固定为指定值，不参与训练
    - 只有未固定的参数和权重w通过MLP学习
    """
    
    def __init__(self, backbone=None, input_size=384, hidden_size=256, K=10, param_share=['mu', 'sigma'], 
                 dropout=0.1, mu_min=0.0, mu_max=200.0, sigma_init=1.0, sigma_min=4.0, fixed_mu=False, fixed_sigma=False, **kwargs):
        super().__init__()
        self.K = K
        self.param_share = param_share
        self.param_names = ['w', 'mu', 'sigma']
        self.params = {}
        self.backbone = backbone
        self.fixed_mu = fixed_mu
        self.fixed_sigma = fixed_sigma
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        
        # 计算需要从MLP输出的参数数量
        # 如果参数被固定，则不需要从MLP输出
        num_trainable_params = 1  # 至少需要输出 w
        if not fixed_mu and 'mu' not in param_share:
            num_trainable_params += 1
        if not fixed_sigma and 'sigma' not in param_share:
            num_trainable_params += 1
        
        # 创建MLP
        self.mlp = MLP(input_size=input_size, hidden_size=hidden_size, 
                      output_size=K * num_trainable_params, dropout_rate=dropout)
        
        # 处理固定参数（注册为buffer，不参与训练）
        if fixed_mu:
            # mu固定为均匀分布
            self.register_buffer('mu_buffer', torch.linspace(mu_min, mu_max, self.K, dtype=torch.float32).unsqueeze(0))
            print(f"Fixed mu mode: mu in [{mu_min}, {mu_max}]")
        elif 'mu' in param_share:
            # mu可训练，作为共享参数
            self.params['mu'] = nn.Parameter(
                torch.linspace(mu_min, mu_max, self.K, dtype=torch.float32).unsqueeze(0))
        
        if fixed_sigma:
            # sigma固定为指定值（加上sigma_min以保持与可训练模式一致）
            self.register_buffer('sigma_buffer', torch.ones((1, self.K)) * (sigma_init + sigma_min))
            print(f"Fixed sigma mode: sigma = {sigma_init + sigma_min} (input: {sigma_init} + {sigma_min})")
        elif 'sigma' in param_share:
            # sigma可训练，作为共享参数
            self.params['sigma'] = nn.Parameter(
                torch.ones((1, self.K)) * inverse_softplus(torch.tensor(sigma_init)))
        
        # 如果有可训练的共享参数，转换为ParameterDict
        if self.params:
            self.params = nn.ParameterDict(self.params)
        
        # 创建变换层（仅当参数可训练时使用）
        if not fixed_mu and 'mu' in param_share:
            self.layer_mu = nn.Sequential(nn.Linear(self.K, self.K))
        if not fixed_sigma and 'sigma' in param_share:
            self.layer_sigma = nn.Sequential(nn.Linear(self.K, self.K))
        
        # 打印配置信息
        if fixed_mu or fixed_sigma:
            print(f"Fixed params mode: fixed_mu={fixed_mu}, fixed_sigma={fixed_sigma}")
        print(f"Sigma min: {sigma_min}")
    
    def cdf(self, params, times):
        ws, mus, sigmas = params['w'], params['mu'], params['sigma']
        device = ws.device
        
        times = times.to(device)
        inv_softplus_times = safe_inverse_softplus(times)
        inv_softplus_times = inv_softplus_times.unsqueeze(-1)
        
        if len(times.shape) == 1:
            normal_dists = torch.distributions.Normal(mus, sigmas)
            repeated_inv_softplus_times = inv_softplus_times.repeat(1, self.K)
            log_normal_cdfs = normal_dists.cdf(repeated_inv_softplus_times)
            cdfs = ws * log_normal_cdfs
            cdfs = cdfs.sum(-1)
        elif len(times.shape) == 2:
            time_len = times.shape[-1]
            mus = mus.unsqueeze(1).repeat(1, time_len, 1)
            sigmas = sigmas.unsqueeze(1).repeat(1, time_len, 1)
            normal_dists = torch.distributions.Normal(mus, sigmas)
            repeated_inv_softplus_times = inv_softplus_times.repeat(1, 1, self.K)
            log_normal_cdfs = normal_dists.cdf(repeated_inv_softplus_times)
            ws = ws.unsqueeze(1)
            cdfs = ws * log_normal_cdfs
            cdfs = cdfs.sum(-1)
            cdfs[cdfs > 1.] = 1.
        else:
            assert False, "times shape not supported"
        return cdfs
    
    def log_prob(self, params, times):
        ws, mus, sigmas = params['w'], params['mu'], params['sigma']
        device = ws.device
        
        times = times.to(device)
        inv_softplus_times = safe_inverse_softplus(times)
        inv_softplus_times = inv_softplus_times.unsqueeze(-1)
        num_components = ws.shape[1]
        normal_dists = torch.distributions.Normal(mus, sigmas)
        repeated_inv_softplus_times = inv_softplus_times.repeat(1, 1, num_components)
        log_normal_pdfs = normal_dists.log_prob(repeated_inv_softplus_times)
        log_pdfs = logsumexp(a=log_normal_pdfs, dim=-1, b=ws).squeeze()
        log_grad = torch.log(inverse_softplus_grad(times))
        return log_pdfs + log_grad
    
    def forward(self, x, **kargs):
        h = self.backbone(x, **kargs) if self.backbone else x
        if isinstance(h, dict):
            subloss = h['loss'] if 'loss' in h else None
            h = h['feat'] if 'feat' in h else h
        else:
            subloss = None
        h = self.mlp(h)
        params = {}
        device = h.device
        i = 0
        
        # 处理权重w（始终从MLP输出）
        params['w'] = h[:, i:i+self.K].softmax(-1)
        i += self.K
        
        # 处理mu
        if self.fixed_mu:
            # mu固定，使用buffer
            params['mu'] = self.mu_buffer.to(device)
        elif 'mu' in self.param_share:
            # mu可训练，作为共享参数
            params['mu'] = self.params['mu']
            if hasattr(self, 'layer_mu'):
                params['mu'] = self.layer_mu(params['mu'])
        else:
            # mu从MLP输出（样本特定）
            params['mu'] = h[:, i:i+self.K]
            i += self.K
        
        # 处理sigma
        if self.fixed_sigma:
            # sigma固定，使用buffer
            params['sigma'] = self.sigma_buffer.to(device)
        elif 'sigma' in self.param_share:
            # sigma可训练，作为共享参数
            params['sigma'] = self.params['sigma']
            if hasattr(self, 'layer_sigma'):
                params['sigma'] = self.layer_sigma(params['sigma'])
            params['sigma'] = F.softplus(params['sigma']) + self.sigma_min
            params['sigma'] = params['sigma'].clamp(min=self.sigma_min, max=1e2)
        else:
            # sigma从MLP输出（样本特定）
            params['sigma'] = h[:, i:i+self.K]
            params['sigma'] = F.softplus(params['sigma']) + self.sigma_min
            params['sigma'] = params['sigma'].clamp(min=self.sigma_min, max=1e2)
        
        # 对mu进行clamp（除非是固定的）
        if not self.fixed_mu:
            params['mu'] = params['mu'].clamp(min=-200, max=200)
        
        return params, subloss
    
    def get_final_params(self):
        """
        Get the final transformed parameters (mu and sigma) that are actually used 
        in the forward pass for computing survival probabilities.
        
        Returns:
            dict with keys 'mu' and 'sigma', each being a torch tensor
        """
        params = {}
        
        # 处理mu
        if self.fixed_mu:
            # mu固定，返回buffer值
            params['mu'] = self.mu_buffer
        elif 'mu' in self.params:
            # mu可训练，作为共享参数
            params['mu'] = self.params['mu']
            if hasattr(self, 'layer_mu'):
                params['mu'] = self.layer_mu(params['mu'])
            params['mu'] = params['mu'].clamp(min=-200, max=200)
        
        # 处理sigma
        if self.fixed_sigma:
            # sigma固定，返回buffer值
            params['sigma'] = self.sigma_buffer
        elif 'sigma' in self.params:
            # sigma可训练，作为共享参数
            params['sigma'] = self.params['sigma']
            if hasattr(self, 'layer_sigma'):
                params['sigma'] = self.layer_sigma(params['sigma'])
            params['sigma'] = F.softplus(params['sigma']) + self.sigma_min
            params['sigma'] = params['sigma'].clamp(min=self.sigma_min, max=1e2)
        
        return params
    
    def train_step(self, x, t, c, constant_dict):
        params, subloss = self(**x)
        survival_func = 1 - self.cdf(params, t)
        pdf = torch.exp(self.log_prob(params, t))
        train_outputs = {
            't': t,
            'c': c,
            'pdf': pdf,
            'survival_func': survival_func,
            'subloss': subloss
        }
        eval_outputs = self.eval_step(x, t, c, constant_dict)
        return train_outputs, eval_outputs
    
    def eval_step(self, x, t, c, constant_dict):
        with torch.no_grad():
            params, _ = self(**x)
            outputs = {}
            # Eval for time-dependent C-index
            outputs["t"] = t
            outputs['c'] = c
            outputs["eval_t"] = constant_dict["eval_t"]  # eval_len
            # 从输入动态获取 batch_size，支持任意 batch_size
            batch_size = x['x'].shape[0]
            device = t.device
            eval_t = constant_dict["eval_t"].repeat(batch_size, 1).to(device)  # batch_size, eval_len
            cdf = self.cdf(params, eval_t)  # batch_size, eval_len
            survival_func = 1 - cdf  # batch_size, eval_len
            cum_hazard = - torch.log(survival_func)
            outputs["cum_hazard_seqs"] = cum_hazard.transpose(1, 0)  # eval_len, batch_size

            # Eval for Brier Score
            t_min = constant_dict["t_min"]
            t_max = constant_dict["t_max"]
            t_seq = torch.linspace(
                t_min.item(), t_max.item(), constant_dict['NUM_INT_STEPS'].item(), dtype=t_min.dtype,
                device=device)
            eval_t = t_seq.unsqueeze(0).repeat(batch_size, 1)  # batch_size, eval_len
            cdf = self.cdf(params, eval_t)  # batch_size, eval_len
            survival_func = 1 - cdf  # batch_size, eval_len
            outputs["survival_seqs"] = survival_func.transpose(1, 0)  # eval_len, batch_size

            for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
                t_min = constant_dict["t_min"]
                t_max = constant_dict["t_max_{}".format(eps)]
                t_seq = torch.linspace(
                    t_min.item(), t_max.item(), constant_dict['NUM_INT_STEPS'].item(), dtype=t_min.dtype,
                    device=device)
                eval_t = t_seq.unsqueeze(0).repeat(batch_size, 1)  # batch_size, eval_len
                cdf = self.cdf(params, eval_t)  # batch_size, eval_len
                survival_func = 1 - cdf  # batch_size, eval_len
                outputs["survival_seqs_{}".format(eps)] = survival_func.transpose(1, 0)  # eval_len, batch_size
        return outputs
    
    def predict_survival(self, x, eval_times=None):
        """
        Predict survival probability for given features
        
        Args:
            x: input features [batch_size, input_size]
            eval_times: evaluation time points [num_time_points]
                       If None, uses default range [0.1, 100] with step 0.1
        
        Returns:
            dict with keys:
                't': time points
                'p_survival': survival probabilities
        """
        device = x.device
        with torch.no_grad():
            if self.backbone:
                params, _ = self(x)
            else:
                # For direct feature input
                params, _ = self.forward(x)
            
            if eval_times is None:
                eval_times = torch.arange(0.1, 100, 0.1).to(device)  #100.1
            else:
                eval_times = eval_times.to(device)
            
            cdf = self.cdf(params, eval_times)
            survival_func = 1 - cdf
        
        return {
            't': eval_times,
            'p_survival': survival_func
        }
    
    def predict_step(self, x):
        device = next(self.parameters()).device
        with torch.no_grad():
            params, _ = self(**x)
            eval_t = torch.arange(0.1, 100, 0.1).to(device)
            cdf = self.cdf(params, eval_t)
            survival_func = 1 - cdf
        ret = {
            't': eval_t,
            'p_survival': survival_func
        }
        return ret