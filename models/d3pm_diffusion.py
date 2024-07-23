# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusion for discrete state spaces."""

import torch
import time
import torch.nn as nn
import torch.nn.functional as F


def make_diffusion(diffusion_config, model_config, num_edges, future_len, device):
    """HParams -> diffusion object."""
    return CategoricalDiffusion(
        betas=get_diffusion_betas(diffusion_config, device),
        model_prediction=model_config['model_prediction'],
        model_output=model_config['model_output'],
        transition_mat_type=model_config['transition_mat_type'],
        transition_bands=model_config['transition_bands'],
        loss_type=model_config['loss_type'],
        hybrid_coeff=model_config['hybrid_coeff'],
        num_edges=num_edges,
        model_name=model_config['name'],
        future_len=future_len,
        device=device
)


def get_diffusion_betas(spec, device):
    """Get betas from the hyperparameters."""
    
    if spec['type'] == 'linear':
        # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
        # To be used with Gaussian diffusion models in continuous and discrete
        # state spaces.
        # To be used with transition_mat_type = 'gaussian'
        return torch.linspace(spec['start'], spec['stop'], spec['num_timesteps']).to(device)
    
    elif spec['type'] == 'cosine':
        # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
        # To be used with transition_mat_type = 'uniform'.
        def cosine_beta_schedule(timesteps, s=0.008):
            """
            Cosine schedule as described in https://arxiv.org/abs/2102.09672.

            Parameters:
            - timesteps: int, the number of timesteps for the schedule.
            - s: float, small constant to prevent numerical issues.

            Returns:
            - betas: torch.Tensor, beta values for each timestep.
            - alphas: torch.Tensor, alpha values for each timestep.
            - alpha_bars: torch.Tensor, cumulative product of alphas for each timestep.
            """
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            alphas = 1 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)

            return betas, alphas, alpha_bars
        betas, alphas, alpha_bars = cosine_beta_schedule(spec['num_timesteps'])
        return betas.to(device)
    
    elif spec['type'] == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
        # To be used with absorbing state models.
        # ensures that the probability of decaying to the absorbing state
        # increases linearly over time, and is 1 for t = T-1 (the final time).
        # To be used with transition_mat_type = 'absorbing'
        return 1. / torch.linspace(spec['num_timesteps'], 1, spec['num_timesteps']).to(device)
    else:
        raise NotImplementedError(spec['type'])


class CategoricalDiffusion:
    """Discrete state space diffusion process.

    Time convention: noisy data is labeled x_0, ..., x_{T-1}, and original data
    is labeled x_start (or x_{-1}). This convention differs from the papers,
    which use x_1, ..., x_T for noisy data and x_0 for original data.
    """

    def __init__(self, *, betas, model_prediction, model_output,
               transition_mat_type, transition_bands, loss_type, hybrid_coeff,
               num_edges, torch_dtype=torch.float32, model_name=None, future_len=None, device=None):

        self.model_prediction = model_prediction  # *x_start*, xprev
        self.model_output = model_output  # logits or *logistic_pars*
        self.loss_type = loss_type  # kl, *hybrid*, cross_entropy_x_start
        self.hybrid_coeff = hybrid_coeff
        self.torch_dtype = torch_dtype
        self.model_name = model_name
        self.device = device

        # Data \in {0, ..., num_edges-1}
        self.num_classes = 2 # 0 or 1
        self.num_edges = num_edges
        self.future_len = future_len
        # self.class_weights = torch.tensor([self.future_len / self.num_edges, 1 - self.future_len / self.num_edges], dtype=torch.float64)
        self.class_weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
        self.class_probs = torch.tensor([1 - self.future_len / self.num_edges, self.future_len / self.num_edges], dtype=torch.float64)
        self.transition_bands = transition_bands
        self.transition_mat_type = transition_mat_type
        self.eps = 1.e-6

        if not isinstance(betas, torch.Tensor):
            raise ValueError('expected betas to be a torch tensor')
        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError('betas must be in (0, 1]')

        # Computations here in float64 for accuracy
        self.betas = betas.to(dtype=torch.float64).to(self.device, non_blocking=True)
        self.num_timesteps, = betas.shape

        # Construct transition matrices for q(x_t|x_{t-1})
        # NOTE: t goes from {0, ..., T-1}
        if self.transition_mat_type == 'uniform':
            q_one_step_mats = [self._get_transition_mat(t) 
                            for t in range(0, self.num_timesteps)]
        elif self.transition_mat_type == 'gaussian':
            q_one_step_mats = [self._get_gaussian_transition_mat(t)
                            for t in range(0, self.num_timesteps)]
        elif self.transition_mat_type == 'absorbing':
            q_one_step_mats = [self._get_absorbing_transition_mat(t)
                            for t in range(0, self.num_timesteps)]
        elif self.transition_mat_type == 'marginal_prior':
            q_one_step_mats = [self._get_prior_distribution_transition_mat(t)
                               for t in range(0, self.num_timesteps)]
        elif self.transition_mat_type == 'custom':
            q_one_step_mats = [self._get_custom_transition_mat(t)
                               for t in range(0, self.num_timesteps)]
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing', 'marginal_prior'"
                f", but is {self.transition_mat_type}"
                )

        self.q_onestep_mats = torch.stack(q_one_step_mats, axis=0).to(self.device, non_blocking=True)
        assert self.q_onestep_mats.shape == (self.num_timesteps,
                                            self.num_classes,
                                            self.num_classes)

        # Construct transition matrices for q(x_t|x_start)
        q_mat_t = self.q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = torch.tensordot(q_mat_t, self.q_onestep_mats[t],
                                    dims=[[1], [0]])
            q_mats.append(q_mat_t)
        self.q_mats = torch.stack(q_mats, axis=0)
        assert self.q_mats.shape == (self.num_timesteps, self.num_classes,
                                    self.num_classes), self.q_mats.shape

        # Don't precompute transition matrices for q(x_{t-1} | x_t, x_start)
        # Can be computed from self.q_mats and self.q_one_step_mats.
        # Only need transpose of q_onestep_mats for posterior computation.
        self.transpose_q_onestep_mats = torch.transpose(self.q_onestep_mats, dim0=1, dim1=2)
        del self.q_onestep_mats

    def _get_full_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Contrary to the band diagonal version, this method constructs a transition
        matrix with uniform probability to all other states.

        Args:
            t: timestep. integer scalar.

        Returns:
            Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        beta_t = self.betas[t]
        # Create a matrix filled with beta_t/num_classes
        mat = torch.full((self.num_classes, self.num_classes), 
                            fill_value=beta_t / float(self.num_classes),
                            dtype=torch.float64)

        # Create a diagonal matrix with values to be set on the diagonal of mat
        diag_val = 1. - beta_t * (self.num_classes - 1.) / self.num_classes
        diag_matrix = torch.diag(torch.full((self.num_classes,), diag_val, dtype=torch.float64))

        # Set the diagonal values
        mat.fill_diagonal_(diag_val)

        return mat

    def _get_transition_mat(self, t):
        r"""Computes transition matrix for q(x_t|x_{t-1}).

        This method constructs a transition
        matrix Q with
        Q_{ij} = beta_t / num_classes       if |i-j| <= self.transition_bands
                1 - \sum_{l \neq i} Q_{il} if i==j.
                0                          else.

        Args:
        t: timestep. integer scalar (or numpy array?)

        Returns:
        Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        if self.transition_bands is None:
            return self._get_full_transition_mat(t)
        # Assumes num_off_diags < num_classes
        beta_t = self.betas[t]
        
        mat = torch.zeros((self.num_classes, self.num_classes),
                        dtype=torch.float64)
        off_diag = torch.full((self.num_classes - 1,), fill_value=beta_t / float(self.num_classes), dtype=torch.float64)

        for k in range(1, self.transition_bands + 1):
            mat += torch.diag(off_diag, k)
            mat += torch.diag(off_diag, -k)
            off_diag = off_diag[:-1]

        # Add diagonal values such that rows sum to one
        diag = 1. - mat.sum(dim=1)
        mat += torch.diag(diag)
        
        return mat

    def _get_gaussian_transition_mat(self, t):
        r"""Computes transition matrix for q(x_t|x_{t-1}).

        This method constructs a transition matrix Q with
        decaying entries as a function of how far off diagonal the entry is.
        Normalization option 1:
        Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                    1 - \sum_{l \neq i} Q_{il}  if i==j.
                    0                          else.

        Normalization option 2:
        tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                            0                        else.

        Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

        Args:
            t: timestep. integer scalar (or numpy array?)

        Returns:
            Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        transition_bands = self.transition_bands if self.transition_bands else self.num_classes - 1

        beta_t = self.betas[t]

        mat = torch.zeros((self.num_classes, self.num_classes),
                        dtype=torch.float64).to(self.device, non_blocking=True)

        # Make the values correspond to a similar type of gaussian as in the
        # gaussian diffusion case for continuous state spaces.
        values = torch.linspace(torch.tensor(0.), torch.tensor(self.num_classes-1), self.num_classes, dtype=torch.float64).to(self.device, non_blocking=True)
        values = values * 2./ (self.num_classes - 1.)
        values = values[:transition_bands+1]
        values = -values * values / beta_t
        
        # To reverse the tensor 'values' starting from the second element
        reversed_values = values[1:].flip(dims=[0])
        # Concatenating the reversed values with the original values
        values = torch.cat([reversed_values, values], dim=0)
        values = F.softmax(values, dim=0)
        values = values[transition_bands:]
        
        for k in range(1, transition_bands + 1):
            off_diag = torch.full((self.num_classes - k,), values[k], dtype=torch.float64).to(self.device, non_blocking=True)

            mat += torch.diag(off_diag, k)
            mat += torch.diag(off_diag, -k)

        # Add diagonal values such that rows and columns sum to one.
        # Technically only the ROWS need to sum to one
        # NOTE: this normalization leads to a doubly stochastic matrix,
        # which is necessary if we want to have a uniform stationary distribution.
        diag = 1. - mat.sum(dim=1)
        mat += torch.diag_embed(diag)

        return mat.to(self.device, non_blocking=True)

    def _get_absorbing_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Has an absorbing state for pixelvalues self.num_classes//2.

        Args:
        t: timestep. integer scalar.

        Returns:
        Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        beta_t = self.betas[t]

        diag = torch.full((self.num_classes,), 1. - beta_t, dtype=torch.float64).to(self.device, non_blocking=True)
        mat = torch.diag(diag)

        # Add beta_t to the num_classes/2-th column for the absorbing state
        mat[:, self.num_classes // 2] += beta_t

        return mat
    
    def _get_prior_distribution_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).
        Use cosine schedule for these transition matrices.

        Args:
        t: timestep. integer scalar.

        Returns:
        Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        beta_t = self.betas[t]
        mat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64).to(self.device, non_blocking=True)

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:
                    mat[i, j] = beta_t * self.class_probs[j]
                else:
                    mat[i, j] = 1 - beta_t + beta_t * self.class_probs[j]
        
        return mat
    
    def _get_custom_transition_mat(self, t):
        """
        Generates a 2x2 transition matrix for a discrete diffusion process at timestep t.

        Parameters:
        - t: int, current timestep.
        - device: torch.device, the device to place the tensor on.

        Returns:
        - transition_matrix: torch.Tensor, a 2x2 transition matrix.
        """
        alphas = 1 - self.betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bar_t = alpha_bars[t].to(self.device)
        
        # Compute the transition probabilities
        transition_matrix = torch.tensor([[alpha_bar_t, 1 - alpha_bar_t],
                                        [1 - alpha_bar_t, alpha_bar_t]], device=self.device)
        
        return transition_matrix

    def _at(self, a, t, x):
        """
        Extract coefficients at specified timesteps t and conditioning data x in PyTorch.

        Args:
        a: torch.Tensor: PyTorch tensor of constants indexed by time, dtype should be pre-set.
        t: torch.Tensor: PyTorch tensor of time indices, shape = (batch_size,).
        x: torch.Tensor: PyTorch tensor of shape (bs, ...) of int32 or int64 type.
            (Noisy) data. Should not be of one-hot representation, but have integer
            values representing the class values. --> NOT A LOT NEEDS TO CHANGE, MY CLASS VALUES ARE SIMPLY 0 AND 1

        Returns:
        a[t, x]: torch.Tensor: PyTorch tensor.
        """
        ### Original ###
        # x.shape = (bs, height, width, channels)
        # t_broadcast_shape = (bs, 1, 1, 1)
        # a.shape = (num_timesteps, num_pixel_vals, num_pixel_vals)
        # out.shape = (bs, height, width, channels, num_pixel_vals)
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        
        ### New ###
        # x.shape = (bs, num_edges, channels=1) 
        # t_broadcast_shape = (bs, 1, 1)
        # a.shape = (num_timesteps, num_classes, num_classes) 
        # out.shape = (bs, num_edges, channels, num_classes) 
        
        # Convert `a` to the desired dtype if not already
        a = a.type(self.torch_dtype)
        #print("A", a.shape)
        

        # Prepare t for broadcasting by adding necessary singleton dimensions
        t_broadcast = t.view(-1, *((1,) * (x.ndim - 1))).to(self.device, non_blocking=True)

        # Advanced indexing in PyTorch to select elements
        #print("_at output", a[t_broadcast, x.long()].shape)
        #print(a[t_broadcast, x.long()])
        return a[t_broadcast, x.long()].to(self.device, non_blocking=True)  # (batch_size, num_edges, 2)

    def _at_onehot(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
        a: torch.Tensor: PyTorch tensor of constants indexed by time, dtype should be pre-set.
        t: torch.Tensor: PyTorch tensor of time indices, shape = (batch_size,).
        x: torch.Tensor: PyTorch tensor of shape (bs, ...) of float32 type.
            (Noisy) data. Should be of one-hot-type representation.

        Returns:
        out: torch.tensor: output of dot(x, a[t], axis=[[-1], [1]]).
            shape = (bs, num_edges, channels=1, num_classes)
        """
        a = a.type(self.torch_dtype)
        
        ### Final ###
        # t.shape = (bs)
        # x.shape = (bs, num_edges, num_classes)
        # a[t].shape = (bs, num_classes, num_classes)
        # out.shape = (bs, num_edges, num_classes)

        a_t = a[t]
        out = torch.einsum('bik,bkj->bij', x, a_t).to(self.device, non_blocking=True)
        
        return out.to(self.device, non_blocking=True)

    def q_probs(self, x_start, t):
        """Compute probabilities of q(x_t | x_start).

        Args:
        x_start: torch.tensor: tensor of shape (bs, ...) of int32 or int64 type.
            Should not be of one hot representation, but have integer values
            representing the class values.
        t: torch.tensor: torch tensor of shape (bs,).

        Returns:
        probs: torch.tensor: shape (bs, x_start.shape[1:],
                                                num_classes).
        """
        return self._at(self.q_mats, t, x_start)

    def q_sample(self, x_start, t, noise):
        """
        Sample from q(x_t | x_start) (i.e. add noise to the data) using Gumbel softmax trick.

        Args:
        x_start: torch.tensor: original clean data, in integer form (not onehot).
            shape = (bs, num_edges).
        t: torch.tensor: timestep of the diffusion process, shape (bs,).
        noise: torch.tensor: uniform noise on [0, 1) used to sample noisy data.
            shape should match (*x_start.shape, num_classes).

        Returns:
        sample: torch.tensor: same shape as x_start. noisy data.
        """
        assert noise.shape == x_start.shape + (self.num_classes,)
        logits = torch.log(self.q_probs(x_start, t) + self.eps)

        # To avoid numerical issues, clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise)).to(self.device, non_blocking=True)
        return torch.argmax(logits + gumbel_noise, dim=-1)
    
    def _get_logits_from_logistic_pars(self, loc, log_scale):
        """
        Computes logits for an underlying logistic distribution.

        Args:
        loc: torch.tensor: location parameter of logistic distribution.
        log_scale: torch.tensor: log scale parameter of logistic distribution.

        Returns:
        logits: torch.tensor: logits corresponding to logistic distribution
        """
        loc = loc.unsqueeze(-1)
        log_scale = log_scale.unsqueeze(-1)

        # Adjust the scale such that if it's zero, the probabilities have a scale
        # that is neither too wide nor too narrow.
        inv_scale = torch.exp(- (log_scale - 2.))

        bin_width = 2. / (self.num_classes - 1.)
        bin_centers = torch.linspace(-1., 1., self.num_classes)

        bin_centers = bin_centers.unsqueeze(0)  # Add batch dimension
        bin_centers = bin_centers - loc

        log_cdf_min = -F.softplus(-inv_scale * (bin_centers - 0.5 * bin_width))
        log_cdf_plus = -F.softplus(-inv_scale * (bin_centers + 0.5 * bin_width))

        logits = torch.log(torch.exp(log_cdf_plus) - torch.exp(log_cdf_min) + self.eps)

        return logits

    def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
        """Compute logits of q(x_{t-1} | x_t, x_start) in PyTorch."""
        
        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.num_classes,), (x_start.shape, x_t.shape)
        else:
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)
            
        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
        if x_start_logits:
            fact2 = self._at_onehot(self.q_mats, t-1, F.softmax(x_start, dim=-1))
            tzero_logits = x_start
        else:
            fact2 = self._at(self.q_mats, t-1, x_start)
            tzero_logits = torch.log(F.one_hot(x_start.to(torch.int64), num_classes=self.num_classes) + self.eps)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.unsqueeze(1).unsqueeze(2)  # Adds new dimensions: [batch_size, 1, 1]
        t_broadcast = t_broadcast.expand(-1, tzero_logits.size(1), tzero_logits.size(-1)).to(self.device, non_blocking=True)   # tzero_logits.size(1) = num_edges, tzero_logits.size(-1) = num_classes

        return torch.where(t_broadcast == 0, tzero_logits, out) # (bs, num_edges, num_classes)

    def p_logits(self, model_fn, x, t, edge_features=None, edge_index=None, condition=None):
        """Compute logits of p(x_{t-1} | x_t) in PyTorch.

        Args:
            model_fn (function): The model function that takes input `x` and `t` and returns the model output.
            x (torch.Tensor): The input tensor of shape (batch_size, num_edges) representing the noised input at time t.
            t (torch.Tensor): The time tensor of shape (batch_size,) representing the time step.

        Returns:
            tuple: A tuple containing two tensors:
                - model_logits (torch.Tensor): The logits of p(x_{t-1} | x_t) of shape (batch_size, num_edges, num_classes).
                - pred_x_start_logits (torch.Tensor): The logits of p(x_{t-1} | x_start) of shape (batch_size, num_edges, num_classes).
        """
        assert t.shape == (x.shape[0],)
        model_output = model_fn(edge_features, edge_index, t, condition=condition)
        #print("Output", model_output.size())

        if self.model_output == 'logits':
            model_logits = model_output
        elif self.model_output == 'logistic_pars':
            loc, log_scale = model_output
            model_logits = self._get_logits_from_logistic_pars(loc, log_scale)
        else:
            raise NotImplementedError(self.model_output)

        if self.model_prediction == 'x_start':
            pred_x_start_logits = model_logits
            #print("Pred_x_start_logits", pred_x_start_logits.shape)
            # print(pred_x_start_logits)
            t_broadcast = t.unsqueeze(1).unsqueeze(2)  # Adds new dimensions: [batch_size, 1, 1]
            t_broadcast = t_broadcast.expand(-1, pred_x_start_logits.size(1), pred_x_start_logits.size(-1)).to(self.device, non_blocking=True)   # pred_x_start_logits.size(1) = num_edges, pred_x_start_logits.size(-1) = num_classes
            model_logits = torch.where(t_broadcast == 0, pred_x_start_logits,
                                       self.q_posterior_logits(x_start=pred_x_start_logits, x_t=x, t=t, x_start_logits=True))
        elif self.model_prediction == 'xprev':
            pred_x_start_logits = model_logits
            raise NotImplementedError(self.model_prediction)
        
        assert (model_logits.shape == pred_x_start_logits.shape == x.shape + (self.num_classes,))
        return model_logits, pred_x_start_logits    # (bs, num_edges, 2)
    
    # === Sampling ===

    def p_sample(self, model_fn, x, t, noise, edge_features=None, edge_index=None, condition=None):
        """Sample one timestep from the model p(x_{t-1} | x_t)."""
        # Get model logits
        model_logits, pred_x_start_logits = self.p_logits(model_fn=model_fn, x=x, t=t, edge_features=edge_features, edge_index=edge_index, condition=condition)

        assert noise.shape == model_logits.shape, noise.shape

        # No noise when t == 0
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *([1] * (len(x.shape) - 1)))
        # For numerical precision clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).eps, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))

        sample = torch.argmax(model_logits + nonzero_mask * gumbel_noise, dim=-1)

        assert sample.shape == x.shape
        assert pred_x_start_logits.shape == model_logits.shape
        
        return sample, pred_x_start_logits

    def p_sample_loop(self, model_fn, shape, num_timesteps=None, return_x_init=False, edge_features=None, edge_index=None, line_graph=None, condition=None, task=None):
        """Ancestral sampling."""
        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.transition_mat_type in ['gaussian', 'uniform', 'marginal_prior', 'custom']:
            # x_init = torch.randint(0, self.num_classes, size=shape, device=device)
            prob_class_1 = 0.5
            x_init = torch.bernoulli(torch.full(size=shape, fill_value=prob_class_1, device=device))
        elif self.transition_mat_type == 'absorbing':
            x_init = torch.full(shape, fill_value=self.num_classes // 2, dtype=torch.int32, device=device)
        else:
            raise ValueError(f"Invalid transition_mat_type {self.transition_mat_type}")

        x = x_init.clone()  # (bs, num_edges)
        edge_attr = x_init.float()
        new_edge_features = edge_features.clone()
        '''for i in range(edge_attr.shape[0]):
            new_edge_features[i * self.num_edges:(i + 1)*self.num_edges, 0] = edge_attr[i]'''
        # TODO: Adapt this for higher batch sizes (see loop above)
        new_edge_features[:, 0] = edge_attr
        
        for i in range(num_timesteps):
            t = torch.full([shape[0]], self.num_timesteps - 1 - i, dtype=torch.long, device=device)
            noise = torch.rand(x.shape + (self.num_classes,), device=device, dtype=torch.float32)
            x, pred_x_start_logits = self.p_sample(model_fn=model_fn, x=x, t=t, noise=noise, edge_features=new_edge_features, edge_index=edge_index, condition=condition)
            new_edge_features[:, 0] = x.float()

        if return_x_init:
            return x_init, x
        else:
            return x

  # === Log likelihood / loss calculation ===
        
    def cross_entropy_x_start(self, x_start, pred_x_start_logits, class_weights):
        """Calculate binary weighted cross entropy between x_start and predicted x_start logits.

        Args:
            x_start (torch.Tensor): original clean data, expected binary labels (0 or 1), shape (bs, num_edges)
            pred_x_start_logits (torch.Tensor): logits as predicted by the model
            class_weights (torch.Tensor): tensor with weights for class 0 and class 1

        Returns:
            torch.Tensor: scalar tensor representing the mean binary weighted cross entropy loss.
        """
        # Calculate binary cross-entropy with logits
        x_start = x_start.long().to(self.device, non_blocking=True)
        pred_x_start_logits = pred_x_start_logits.permute(0, 2, 1).float() # (bs, num_edges, num_classes) -> (bs, num_classes, num_edges)
        ce = F.cross_entropy(pred_x_start_logits, x_start, weight=class_weights.float().to(self.device, non_blocking=True), reduction='mean')

        return ce

    def training_losses(self, model_fn, condition=None, *, x_start, edge_features, edge_index, line_graph=None):
        """Training loss calculation."""
        # Add noise to data
        batch_size, num_edges = x_start.shape
        noise = torch.rand(x_start.shape + (self.num_classes,), dtype=torch.float32)
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],))

        # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint itself.
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)  # (bs, num_edges)
        #print("X_start", [torch.argwhere(x_start[i].unsqueeze(0) == 1)[:, 1] for i in range(x_start.shape[0])])
        #print("X_t", [torch.argwhere(x_t[i].unsqueeze(0) == 1)[:, 1] for i in range(x_t.shape[0])])
        #print("t", t)
        #print("\n")
        
        edge_attr_t = x_t.float()   # (bs, num_edges)
        new_edge_features = edge_features.clone()
        for i in range(edge_attr_t.shape[0]):
            new_edge_features[i * num_edges:(i + 1)*num_edges, 0] = edge_attr_t[i]


        # Calculate the loss
        if self.loss_type == 'kl':
            losses, pred_x_start_logits = self.vb_terms_bpd(model_fn=model_fn, x_start=x_start, x_t=x_t, t=t,
                                                               edge_features=new_edge_features, edge_index=edge_index, condition=condition)
            
            pred_x_start_logits = pred_x_start_logits.squeeze(2)    # (bs, num_edges, channels, classes) -> (bs, num_edges, classes)
            # NOTE: Currently only works for batch size of 1
            pred_x_start_logits = pred_x_start_logits.squeeze(0)    # (bs, num_edges, classes) -> (num_edges, classes)
            pred = pred_x_start_logits.argmax(dim=1)                # (num_edges, classes) -> (num_edges,)
            
            return losses, pred
            
        elif self.loss_type == 'cross_entropy_x_start':
            
            _, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t, edge_features=new_edge_features, edge_index=edge_index, condition=condition)
            losses = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits, class_weights=self.class_weights)
            
            pred_x_start_logits = pred_x_start_logits.squeeze(2)    # (bs, num_edges, channels, classes) -> (bs, num_edges, classes)
            
            if (self.model_name == 'edge_encoder') | (self.model_name == 'edge_encoder_residual'):
                # NOTE: Currently only works for batch size of 1
                pred_x_start_logits = pred_x_start_logits.squeeze(0)    # (bs, num_edges, classes) -> (num_edges, classes)
                pred = pred_x_start_logits.argmax(dim=2)    # (num_edges, classes) -> (num_edges,)
            elif self.model_name == 'edge_encoder_mlp':
                pred = pred_x_start_logits.argmax(dim=2)
            
            return losses, pred
            
        else:
            raise NotImplementedError(self.loss_type)

        return losses
