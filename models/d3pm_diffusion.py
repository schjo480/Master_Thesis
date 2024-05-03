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

from utils.d3pm_utils import categorical_kl_logits, meanflat, categorical_log_likelihood, categorical_kl_probs
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_diffusion(diffusion_config, model_config, num_edges):
    """HParams -> diffusion object."""
    return CategoricalDiffusion(
        betas=get_diffusion_betas(diffusion_config),
        model_prediction=model_config['model_prediction'],
        model_output=model_config['model_output'],
        transition_mat_type=model_config['transition_mat_type'],
        transition_bands=model_config['transition_bands'],
        loss_type=model_config['loss_type'],
        hybrid_coeff=model_config['hybrid_coeff'],
        num_edges=num_edges)


def get_diffusion_betas(spec):
    """Get betas from the hyperparameters."""
    
    if spec['type'] == 'linear':
        # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
        # To be used with Gaussian diffusion models in continuous and discrete
        # state spaces.
        # To be used with transition_mat_type = 'gaussian'
        return torch.linspace(spec['start'], spec['stop'], spec['num_timesteps'])
    elif spec['type'] == 'cosine':
        # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
        # To be used with transition_mat_type = 'uniform'.
        steps = torch.linspace(0, 1, spec['num_timesteps'] + 1, dtype=torch.float64)
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        betas = torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.tensor(0.999))
        return betas
    elif spec['type'] == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
        # To be used with absorbing state models.
        # ensures that the probability of decaying to the absorbing state
        # increases linearly over time, and is 1 for t = T-1 (the final time).
        # To be used with transition_mat_type = 'absorbing'
        return 1. / torch.linspace(spec['num_timesteps'], 1, spec['num_timesteps'])
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
               num_edges, torch_dtype=torch.float32):

        self.model_prediction = model_prediction  # *x_start*, xprev
        self.model_output = model_output  # logits or *logistic_pars*
        self.loss_type = loss_type  # kl, *hybrid*, cross_entropy_x_start
        self.hybrid_coeff = hybrid_coeff
        self.torch_dtype = torch_dtype

        # Data \in {0, ..., num_edges-1}
        self.num_edges = num_edges
        self.transition_bands = transition_bands
        self.transition_mat_type = transition_mat_type
        self.eps = 1.e-6

        if not isinstance(betas, torch.Tensor):
            raise ValueError('expected betas to be a numpy array')
        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError('betas must be in (0, 1]')

        # Computations here in float64 for accuracy
        self.betas = betas = betas.to(dtype=torch.float64)
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
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
                f", but is {self.transition_mat_type}"
                )

        self.q_onestep_mats = torch.stack(q_one_step_mats, axis=0)
        assert self.q_onestep_mats.shape == (self.num_timesteps,
                                            self.num_edges,
                                            self.num_edges)

        # Construct transition matrices for q(x_t|x_start)
        q_mat_t = self.q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = torch.tensordot(q_mat_t, self.q_onestep_mats[t],
                                    dims=[[1], [0]])
            q_mats.append(q_mat_t)
        self.q_mats = torch.stack(q_mats, axis=0)
        assert self.q_mats.shape == (self.num_timesteps, self.num_edges,
                                    self.num_edges), self.q_mats.shape

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
            Q_t: transition matrix. shape = (num_edges, num_edges).
        """
        beta_t = self.betas[t]
        # Create a matrix filled with beta_t/num_edges
        mat = torch.full((self.num_edges, self.num_edges), 
                            fill_value=beta_t / float(self.num_edges),
                            dtype=torch.float64)

        # Create a diagonal matrix with values to be set on the diagonal of mat
        diag_val = 1. - beta_t * (self.num_edges - 1.) / self.num_edges
        diag_matrix = torch.diag(torch.full((self.num_edges,), diag_val, dtype=torch.float64))

        # Set the diagonal values
        mat.fill_diagonal_(diag_val)

        return mat

    def _get_transition_mat(self, t):
        r"""Computes transition matrix for q(x_t|x_{t-1}).

        This method constructs a transition
        matrix Q with
        Q_{ij} = beta_t / num_edges       if |i-j| <= self.transition_bands
                1 - \sum_{l \neq i} Q_{il} if i==j.
                0                          else.

        Args:
        t: timestep. integer scalar (or numpy array?)

        Returns:
        Q_t: transition matrix. shape = (num_edges, num_edges).
        """
        if self.transition_bands is None:
            return self._get_full_transition_mat(t)
        # Assumes num_off_diags < num_edges
        beta_t = self.betas[t]
        
        mat = torch.zeros((self.num_edges, self.num_edges),
                        dtype=torch.float64)
        off_diag = torch.full((self.num_edges - 1,), fill_value=beta_t / float(self.num_edges), dtype=torch.float64)

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
            Q_t: transition matrix. shape = (num_edges, num_edges).
        """
        transition_bands = self.transition_bands if self.transition_bands else self.num_edges - 1

        beta_t = self.betas[t]

        mat = torch.zeros((self.num_edges, self.num_edges),
                        dtype=torch.float64)

        # Make the values correspond to a similar type of gaussian as in the
        # gaussian diffusion case for continuous state spaces.
        values = torch.linspace(torch.tensor(0.), torch.tensor(self.num_edges-1), self.num_edges, dtype=torch.float64)
        values = values * 2./ (self.num_edges - 1.)
        values = values[:transition_bands+1]
        values = -values * values / beta_t
        
        # To reverse the tensor 'values' starting from the second element
        reversed_values = values[1:].flip(dims=[0])
        # Concatenating the reversed values with the original values
        values = torch.cat([reversed_values, values], dim=0)
        values = F.softmax(values, dim=0)
        values = values[transition_bands:]
        
        for k in range(1, transition_bands + 1):
            off_diag = torch.full((self.num_edges - k,), values[k], dtype=torch.float64)

            mat += torch.diag(off_diag, k)
            mat += torch.diag(off_diag, -k)

        # Add diagonal values such that rows and columns sum to one.
        # Technically only the ROWS need to sum to one
        # NOTE: this normalization leads to a doubly stochastic matrix,
        # which is necessary if we want to have a uniform stationary distribution.
        diag = 1. - mat.sum(dim=1)
        mat += torch.diag_embed(diag)

        return mat

    def _get_absorbing_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Has an absorbing state for pixelvalues self.num_edges//2.

        Args:
        t: timestep. integer scalar.

        Returns:
        Q_t: transition matrix. shape = (num_edges, num_edges).
        """
        beta_t = self.betas[t]

        diag = torch.full((self.num_edges,), 1. - beta_t, dtype=torch.float64)
        mat = torch.diag(diag)

        # Add beta_t to the num_edges/2-th column for the absorbing state
        mat[:, self.num_edges // 2] += beta_t

        return mat

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
        # x.shape = (bs, future_len, channels=1) --> OK
        # t_broadcast_shape = (bs, 1, 1)
        # a.shape = (num_timesteps, num_edges, num_edges) --> OK
        # out.shape = (bs, future_len, channels, num_edges) --> OK
        
        # Convert `a` to the desired dtype if not already
        a = a.type(self.torch_dtype)

        # Prepare t for broadcasting by adding necessary singleton dimensions
        t_broadcast = t.view(-1, *((1,) * (x.ndim - 1)))

        # Advanced indexing in PyTorch to select elements
        return a[t_broadcast, x]

    def _at_onehot(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
        a: torch.Tensor: PyTorch tensor of constants indexed by time, dtype should be pre-set.
        t: torch.Tensor: PyTorch tensor of time indices, shape = (batch_size,).
        x: torch.Tensor: PyTorch tensor of shape (bs, ...) of float32 type.
            (Noisy) data. Should be of one-hot-type representation.

        Returns:
        out: torch.tensor: Jax array. output of dot(x, a[t], axis=[[-1], [1]]).
            shape = (bs, ..., num_edges)
        """
        a = a.type(self.torch_dtype)

        ### New ###
        # t.shape = (bs)
        # x.shape = (bs, future_len, channels=1, num_edges)
        # a[t].hape = (bs, num_edges, num_edges)
        # out.shape = (bs, future_len, channels=1, num_edges)

        a_t = a[t]
        out = torch.einsum('bijc,bjk->bik', x, a_t) # Multiply last dimension of x with last 2 dimensions of a_t
        out = out.unsqueeze(2) # Add channel dimension
        
        return out

    def q_probs(self, x_start, t):
        """Compute probabilities of q(x_t | x_start).

        Args:
        x_start: torch.tensor: tensor of shape (bs, ...) of int32 or int64 type.
            Should not be of one hot representation, but have integer values
            representing the class values.
        t: torch.tensor: jax array of shape (bs,).

        Returns:
        probs: torch.tensor: shape (bs, x_start.shape[1:],
                                                num_edges).
        """
        return self._at(self.q_mats, t, x_start)

    def q_sample(self, x_start, t, noise):
        """
        Sample from q(x_t | x_start) (i.e. add noise to the data) using Gumbel softmax trick.

        Args:
        x_start: torch.tensor: original clean data, in integer form (not onehot).
            shape = (bs, ...).
        t: torch.tensor: timestep of the diffusion process, shape (bs,).
        noise: torch.tensor: uniform noise on [0, 1) used to sample noisy data.
            shape should match (*x_start.shape, num_edges).

        Returns:
        sample: torch.tensor: same shape as x_start. noisy data.
        """
        assert noise.shape == x_start.shape + (self.num_edges,)
        logits = torch.log(self.q_probs(x_start, t) + self.eps)

        # To avoid numerical issues, clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
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

        bin_width = 2. / (self.num_edges - 1.)
        bin_centers = torch.linspace(-1., 1., self.num_edges)

        bin_centers = bin_centers.unsqueeze(0)  # Add batch dimension
        bin_centers = bin_centers - loc

        log_cdf_min = -F.softplus(-inv_scale * (bin_centers - 0.5 * bin_width))
        log_cdf_plus = -F.softplus(-inv_scale * (bin_centers + 0.5 * bin_width))

        logits = torch.log(torch.exp(log_cdf_plus) - torch.exp(log_cdf_min) + self.eps)

        return logits

    def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
        """Compute logits of q(x_{t-1} | x_t, x_start) in PyTorch."""
        
        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.num_edges,), (x_start.shape, x_t.shape)
        else:
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
        if x_start_logits:
            fact2 = self._at_onehot(self.q_mats, t-1, F.softmax(x_start, dim=-1))
            tzero_logits = x_start
        else:
            fact2 = self._at(self.q_mats, t-1, x_start)
            tzero_logits = torch.log(F.one_hot(x_start.to(torch.int64), num_classes=self.num_edges) + self.eps)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Adds new dimensions: [batch_size, 1, 1, 1]
        t_broadcast = t_broadcast.expand(-1, tzero_logits.size(1), 1, tzero_logits.size(-1))   # tzero_logits.size(1) = future_len, tzero_logits.size(-1) = num_edges
        
        return torch.where(t_broadcast == 0, tzero_logits, out) # (bs, future_len, channels=1, num_edges)

    def p_logits(self, model_fn, x, t, node_features=None, edge_index=None, edge_attr=None, condition=None):
        """Compute logits of p(x_{t-1} | x_t) in PyTorch.

        Args:
            model_fn (function): The model function that takes input `x` and `t` and returns the model output.
            x (torch.Tensor): The input tensor of shape (batch_size, input_size) representing the noised input at time t.
            t (torch.Tensor): The time tensor of shape (batch_size,) representing the time step.

        Returns:
            tuple: A tuple containing two tensors:
                - model_logits (torch.Tensor): The logits of p(x_{t-1} | x_t) of shape (batch_size, input_size, num_edges).
                - pred_x_start_logits (torch.Tensor): The logits of p(x_{t-1} | x_start) of shape (batch_size, input_size, num_edges).
        """
        assert t.shape == (x.shape[0],)
        model_output = model_fn(node_features, edge_index, t, edge_attr, condition=condition)

        if self.model_output == 'logits':
            model_logits = model_output
        elif self.model_output == 'logistic_pars':
            loc, log_scale = model_output
            model_logits = self._get_logits_from_logistic_pars(loc, log_scale)
        else:
            raise NotImplementedError(self.model_output)

        if self.model_prediction == 'x_start':
            pred_x_start_logits = model_logits
            t_broadcast = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Adds new dimensions: [batch_size, 1, 1, 1]
            t_broadcast = t_broadcast.expand(-1, pred_x_start_logits.size(1), 1, pred_x_start_logits.size(-1))   # pred_x_start_logits.size(1) = future_len, pred_x_start_logits.size(-1) = num_edges

            model_logits = torch.where(t_broadcast == 0, pred_x_start_logits,
                                       self.q_posterior_logits(pred_x_start_logits, x, t, x_start_logits=True))
        elif self.model_prediction == 'xprev':
            pred_x_start_logits = model_logits
            raise NotImplementedError(self.model_prediction)

        assert (model_logits.shape == pred_x_start_logits.shape == x.shape + (self.num_edges,))
        return model_logits, pred_x_start_logits
    
    # === Sampling ===

    def p_sample(self, model_fn, x, t, noise, node_features=None, edge_index=None, edge_attr=None, condition=None):
        """Sample one timestep from the model p(x_{t-1} | x_t)."""
        model_logits, pred_x_start_logits = self.p_logits(model_fn=model_fn, x=x, t=t, node_features=node_features, edge_index=edge_index, edge_attr=edge_attr, condition=condition)
        assert noise.shape == model_logits.shape, noise.shape

        # No noise when t == 0
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *([1] * (len(x.shape) - 1)))
        # For numerical precision clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).eps, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))

        sample = torch.argmax(model_logits + nonzero_mask * gumbel_noise, dim=-1)

        assert sample.shape == x.shape
        assert pred_x_start_logits.shape == model_logits.shape
        return sample, F.softmax(pred_x_start_logits, dim=-1)

    def p_sample_loop(self, model_fn, shape, num_timesteps=None, return_x_init=False, node_features=None, edge_index=None, edge_attr=None, condition=None):
        """Ancestral sampling."""
        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.transition_mat_type in ['gaussian', 'uniform']:
            x_init = torch.randint(0, self.num_edges, size=shape, device=device)
        elif self.transition_mat_type == 'absorbing':
            x_init = torch.full(shape, fill_value=self.num_edges // 2, dtype=torch.int32, device=device)
        else:
            raise ValueError(f"Invalid transition_mat_type {self.transition_mat_type}")

        x = x_init.clone()
        for i in range(num_timesteps):
            t = torch.full([shape[0]], self.num_timesteps - 1 - i, dtype=torch.long, device=device)
            noise = torch.rand(x.shape + (self.num_edges,), device=device)
            x, _ = self.p_sample(model_fn=model_fn, x=x, t=t, noise=noise, node_features=node_features, edge_index=edge_index, edge_attr=edge_attr, condition=condition)

        if return_x_init:
            return x_init, x
        else:
            return x

  # === Log likelihood / loss calculation ===

    def vb_terms_bpd(self, model_fn, *, x_start, x_t, t, node_features=None, edge_index=None, edge_attr=None, condition=None):
        """Calculate specified terms of the variational bound.

        Args:
        model_fn: the denoising network
        x_start: original clean data
        x_t: noisy data
        t: timestep of the noisy data (and the corresponding term of the bound
            to return)

        Returns:
        a pair `(kl, pred_start_logits)`, where `kl` are the requested bound terms
        (specified by `t`), and `pred_x_start_logits` is logits of
        the denoised image.
        """
        true_logits = self.q_posterior_logits(x_start, x_t, t, x_start_logits=False)
        model_logits, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t, node_features=node_features, edge_index=edge_index, edge_attr=edge_attr, condition=condition)

        kl = categorical_kl_logits(logits1=true_logits, logits2=model_logits)
        assert kl.shape == x_start.shape
        kl = meanflat(kl) / torch.log(torch.tensor(2.0))

        decoder_nll = -categorical_log_likelihood(x_start, model_logits)
        assert decoder_nll.shape == x_start.shape
        decoder_nll = meanflat(decoder_nll) / torch.log(torch.tensor(2.0))

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_start) || p(x_{t-1}|x_t))
        assert kl.shape == decoder_nll.shape == t.shape == (x_start.shape[0],)
        result = torch.where(t == 0, decoder_nll, kl)
        return result, pred_x_start_logits

    def prior_bpd(self, x_start):
        """KL(q(x_{T-1}|x_start)|| U(x_{T-1}|0, num_edges-1))."""
        q_probs = self.q_probs(
            x_start=x_start,
            t=torch.full((x_start.shape[0],), self.num_timesteps - 1, dtype=torch.long))

        if self.transition_mat_type in ['gaussian', 'uniform']:
            # Stationary distribution is a uniform distribution over all pixel values.
            prior_probs = torch.ones_like(q_probs) / self.num_edges
        elif self.transition_mat_type == 'absorbing':
            absorbing_int = torch.full(x_start.shape[:-1], self.num_edges // 2, dtype=torch.int32)
            prior_probs = F.one_hot(absorbing_int, num_classes=self.num_edges).to(dtype=self.torch_dtype)
        else:
            raise ValueError("Invalid transition_mat_type")


        assert prior_probs.shape == q_probs.shape

        kl_prior = categorical_kl_probs(
            q_probs, prior_probs)
        assert kl_prior.shape == x_start.shape
        return meanflat(kl_prior) / torch.log(torch.tensor(2.0))

    def cross_entropy_x_start(self, x_start, pred_x_start_logits):
        """Calculate crossentropy between x_start and predicted x_start.

        Args:
        x_start: original clean data
        pred_x_start_logits: predicted_logits

        Returns:
        ce: cross entropy.
        """

        ce = -categorical_log_likelihood(x_start, pred_x_start_logits)
        assert ce.shape == x_start.shape
        ce = meanflat(ce) / torch.log(torch.tensor(2.0))

        assert ce.shape == (x_start.shape[0],)

        return ce

    def training_losses(self, model_fn, node_features=None, edge_index=None, data=None, condition=None, *, x_start):
        """Training loss calculation."""
        # Add noise to data
        # TODO: Reshape x_start to be [batch_size, future_len, channels=1]
        x_start = x_start.unsqueeze(-1)  # Adds a singleton dimension at the last position
        noise = torch.rand(x_start.shape + (self.num_edges,), dtype=torch.float32)
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],))

        # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint itself.
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Initialize the tensor to store edge attributes for all items in the batch
        edge_attr_t = torch.zeros(x_t.size(0), self.num_edges, 1, dtype=torch.float32)

        # Iterate over each item in the batch to set the appropriate indices to 1
        for idx, edges in enumerate(x_t):
            # Set the specific indices for the current batch item
            # Since we are setting 1s, we'll use the indices directly within our 3D tensor
            edge_attr_t[idx, edges, 0] = 1.
            # Generate and set the second attribute to either -1 or 1 randomly
            # edge_attr_t[idx, edges, 1] = 1. if torch.rand(1) > 0.5 else -1

        # Calculate the loss
        if self.loss_type == 'kl':
            losses, _ = self.vb_terms_bpd(model_fn=model_fn, x_start=x_start, x_t=x_t, t=t)
            
        elif self.loss_type == 'cross_entropy_x_start':
            _, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t, node_features=node_features, edge_index=edge_index, edge_attr=edge_attr_t, condition=condition)
            
            losses = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits)
            
        elif self.loss_type == 'hybrid':
            vb_losses, pred_x_start_logits = self.vb_terms_bpd(model_fn=model_fn, x_start=x_start, x_t=x_t, t=t,
                                                               node_features=node_features, edge_index=edge_index, edge_attr=edge_attr_t, condition=condition)
            ce_losses = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits)
            losses = vb_losses + self.hybrid_coeff * ce_losses
            
        else:
            raise NotImplementedError(self.loss_type)

        return losses

    def calc_bpd_loop(self, model_fn, *, x_start, rng):
        """Calculate variational bound (loop over all timesteps and sum)."""
        batch_size = x_start.shape[0]
        total_vb = torch.zeros(batch_size)

        for t in range(self.num_timesteps):
            noise = torch.rand(x_start.shape + (self.num_edges,), dtype=torch.float32)
            x_t = self.q_sample(x_start=x_start, t=torch.full((batch_size,), t), noise=noise)
            vb, _ = self.vb_terms_bpd(model_fn=model_fn, x_start=x_start, x_t=x_t, t=torch.full((batch_size,), t))
            total_vb += vb

        prior_b = self.prior_bpd(x_start=x_start)
        total_b = total_vb + prior_b

        return {
            'total': total_b,
            'vbterms': total_vb,
            'prior': prior_b
        }
