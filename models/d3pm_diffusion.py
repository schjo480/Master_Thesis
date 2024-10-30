"""Diffusion for discrete state spaces. 
Code adapted from D3PM codebase: https://github.com/google-research/google-research/tree/master/d3pm/images"""

import torch
import time
import torch.nn as nn
import torch.nn.functional as F

def make_diffusion(diffusion_config, model_config, num_edges, future_len, edge_features, device, avg_future_len=None):
    """HParams -> diffusion object."""
    return CategoricalDiffusion(
        betas=get_diffusion_betas(diffusion_config, device),
        transition_mat_type=model_config['transition_mat_type'],
        num_edges=num_edges,
        model_name=model_config['name'],
        future_len=future_len,
        edge_features=edge_features,
        device=device,
        avg_future_len=avg_future_len
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

    def __init__(self, *, betas,
               transition_mat_type, num_edges, torch_dtype=torch.float32, 
               model_name=None, future_len=None, edge_features=None, device=None, avg_future_len=None):

        self.torch_dtype = torch_dtype
        self.model_name = model_name
        self.device = device

        # Data \in {0, ..., num_edges-1}
        self.num_classes = 2 # 0 or 1
        self.num_edges = num_edges
        if future_len > 0:
            self.future_len = future_len
        else:
            self.future_len = avg_future_len
        self.edge_features = edge_features
        # self.class_weights = torch.tensor([self.future_len / self.num_edges, 1 - self.future_len / self.num_edges], dtype=torch.float64)
        self.class_weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
        self.class_probs = torch.tensor([1 - self.future_len / self.num_edges, self.future_len / self.num_edges], dtype=torch.float64)
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
        if self.transition_mat_type == 'marginal_prior':
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

    # === Transition matrices ===

    def _get_prior_distribution_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).
        The prior distribution is the distribution of the data at the start of the diffusion process.
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
        """Computes transition matrix for q(x_t|x_{t-1}).
        Produces uniform transition matrices.
        Args:
        - t: int, current timestep.

        Returns:
        Q_t: transition matrix. torch.Tensor, a 2x2 transition matrix.
        """
        alphas = 1 - self.betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bar_t = alpha_bars[t].to(self.device)
        
        # Compute the transition probabilities
        transition_matrix = torch.tensor([[alphas[t], 1 - alphas[t]],
                                        [1 - alphas[t], alphas[t]]], device=self.device)
        
        return transition_matrix

    def _at(self, q_mats, t, x):
        """
        Extract coefficients at specified timesteps t and conditioning data x in PyTorch.

        Args:
        q_mats: torch.Tensor: PyTorch tensor of constants indexed by time, dtype should be pre-set.
        t: torch.Tensor: PyTorch tensor of time indices, shape = (batch_size,).
        x: torch.Tensor: PyTorch tensor of shape (bs, ...) of int32 or int64 type.
            (Noisy) data. Should not be of one-hot representation, but have integer
            values representing the class values.

        Returns:
        q_mat[t, x]: torch.Tensor
        """
        # x.shape = (bs, num_edges) 
        # t_broadcast_shape = (bs, 1)
        # q_mats.shape = (num_timesteps, num_classes, num_classes) 
        # out.shape = (bs, num_edges, num_classes) 
        
        # Convert `q_mats` to the desired dtype if not already
        q_mats = q_mats.type(self.torch_dtype)        

        # Prepare t for broadcasting by adding necessary singleton dimensions
        t_broadcast = t.unsqueeze(1).to(self.device, non_blocking=True)

        # Advanced indexing in PyTorch to select elements
        return q_mats[t_broadcast, x.long()].to(self.device, non_blocking=True)  # (batch_size, num_edges, 2)

    def _at_onehot(self, q_mat, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
        q_mat: torch.Tensor: PyTorch tensor of constants indexed by time, dtype should be pre-set.
        t: torch.Tensor: PyTorch tensor of time indices, shape = (batch_size,).
        x: torch.Tensor: PyTorch tensor of shape (bs, ...) of float32 type.
            (Noisy) data. Should be of one-hot-type representation.

        Returns:
        out: torch.tensor: output of dot(x, q_mat[t], axis=[[-1], [1]]).
            shape = (bs, num_edges, channels=1, num_classes)
        """
        q_mat = q_mat.type(self.torch_dtype)
        
        ### Final ###
        # t.shape = (bs)
        # x.shape = (bs, num_edges, num_classes)
        # q_mat[t].shape = (bs, num_classes, num_classes)
        # out.shape = (bs, num_edges, num_classes)

        q_mat_t = q_mat[t]
        out = torch.einsum('bik,bkj->bij', x, q_mat_t).to(self.device, non_blocking=True)
        
        return out.to(self.device, non_blocking=True)

    # === Noising and Logits calculation ===
    
    def q_probs(self, x_start, t):
        """Compute probabilities of q(x_t | x_start).

        Args:
        x_start: torch.tensor: tensor of shape (bs, ...) of int32 or int64 type.
            Should not be of one hot representation, but have integer values
            representing the class values.
        t: torch.tensor: torch tensor of shape (bs,).

        Returns:
        probs: torch.tensor: shape (batch_size, num_edges, 2).
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
            shape should match (x_start.shape, num_classes).

        Returns:
        sample: torch.tensor: same shape as x_start. noisy data.
        """
        assert noise.shape == x_start.shape + (self.num_classes,)
        logits = torch.log(self.q_probs(x_start, t) + self.eps) # (bs, num_edges, num_classes)

        # To avoid numerical issues, clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)   # (bs, num_edges, num_classes)
        gumbel_noise = -torch.log(-torch.log(noise)).to(self.device, non_blocking=True) # (bs, num_edges, num_classes)
        
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
        """Compute logits of q(x_{t-1} | x_t, x_start_tilde) in PyTorch."""
        
        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.num_classes,), (x_start.shape, x_t.shape)
        else:
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)
        
        # fact1 = x_t * Q_t.T
        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t) # (batch_size, num_edges, num_classes)
        if x_start_logits:
            # x_start represents the logits of x_start
            # F.softmax(x_start, dim=-1) represents x_0_tilde from the D3PM paper, or x_start_tilde
            # fact2 = x_start_tilde * Q_{t-1}_bar
            fact2 = self._at_onehot(self.q_mats, t-1, F.softmax(x_start, dim=-1))   # (batch_size, num_edges, num_classes)
            tzero_logits = x_start
        else:
            fact2 = self._at(self.q_mats, t-1, x_start)
            tzero_logits = torch.log(F.one_hot(x_start.to(torch.int64), num_classes=self.num_classes) + self.eps)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.unsqueeze(1).unsqueeze(2)  # Adds new dimensions: [batch_size, 1, 1]
        t_broadcast = t_broadcast.expand(-1, tzero_logits.size(1), tzero_logits.size(-1)).to(self.device, non_blocking=True)   # tzero_logits.size(1) = num_edges, tzero_logits.size(-1) = num_classes

        return torch.where(t_broadcast == 0, tzero_logits, out) # (bs, num_edges, num_classes)

    def p_logits(self, model_fn, x, t, edge_features=None, edge_index=None, indices=None):
        """Compute logits of p(x_{t-1} | x_t) in PyTorch.
        p(x_{t-1}|x_t) ~ sum_over_x_start_tilde(q(x_{t-1}, x_t | x_start_tilde) * p(x_start_tilde|x_t))
        
        with q(x_{t-1}, x_t | x_start_tilde) ~ q(x_{t-1} | x_t, x_start_tilde) * q(x_t | x_start_tilde)
            where q(x_{t-1} | x_t, x_start_tilde) is the q_posterior_logits

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
        model_output = model_fn(edge_features, edge_index, t, indices)

        model_logits = model_output

        pred_x_start_logits = model_logits
        t_broadcast = t.unsqueeze(1).unsqueeze(2)  # Adds new dimensions: [batch_size, 1, 1]
        t_broadcast = t_broadcast.expand(-1, pred_x_start_logits.size(1), pred_x_start_logits.size(-1)).to(self.device, non_blocking=True)   # pred_x_start_logits.size(1) = num_edges, pred_x_start_logits.size(-1) = num_classes
        model_logits = torch.where(t_broadcast == 0, pred_x_start_logits,
                                    self.q_posterior_logits(x_start=pred_x_start_logits, x_t=x, t=t, x_start_logits=True))
    
        assert (model_logits.shape == pred_x_start_logits.shape == x.shape + (self.num_classes,))
        return model_logits, pred_x_start_logits    # (bs, num_edges, 2)
    
    # === Sampling ===

    def p_sample(self, model_fn, x, t, noise, edge_features=None, edge_index=None, indices=None):
        """Sample one timestep from the model p(x_{t-1} | x_t)."""
        # Get model logits
        model_logits, pred_x_start_logits = self.p_logits(model_fn=model_fn, x=x, t=t, edge_features=edge_features, edge_index=edge_index, indices=indices)

        assert noise.shape == model_logits.shape, noise.shape

        # For numerical precision clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).eps, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        
        # No noise when t == 0
        if t[0] == 0:
            sample = torch.argmax(model_logits, dim=-1)
        else:
            sample = torch.argmax(model_logits + gumbel_noise, dim=-1)

        assert sample.shape == x.shape
        assert pred_x_start_logits.shape == model_logits.shape
        
        return sample, pred_x_start_logits

    def p_sample_loop(self, model_fn, shape, num_timesteps=None, return_x_init=False, edge_features=None, edge_index=None, indices=None, task=None):
        """Ancestral sampling."""
        num_edges = shape[1]
        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.transition_mat_type in ['gaussian', 'uniform', 'custom']:
            # x_init = torch.randint(0, self.num_classes, size=shape, device=device)
            prob_class_1 = 0.5
            x_init = torch.bernoulli(torch.full(size=shape, fill_value=prob_class_1, device=device))
        elif self.transition_mat_type == 'marginal_prior':
            prob_class_1 = self.future_len / self.num_edges
            x_init = torch.bernoulli(torch.full(size=shape, fill_value=prob_class_1, device=device))
        elif self.transition_mat_type == 'absorbing':
            x_init = torch.full(shape, fill_value=self.num_classes // 2, dtype=torch.int32, device=device)
        else:
            raise ValueError(f"Invalid transition_mat_type {self.transition_mat_type}")
        
        x = x_init.clone()  # (bs, num_edges)
        edge_attr = x_init.float()
        new_edge_features = edge_features.clone()
        new_edge_features[:, -1] = edge_attr.flatten()
        if 'num_pred_edges' in self.edge_features:
            new_edge_features = torch.cat((new_edge_features, torch.zeros((new_edge_features.size(0), 1), device=new_edge_features.device)), dim=1)
            new_edge_features[:, -1] = torch.sum(x, dim=1).repeat_interleave(self.num_edges) / self.num_edges
        
        for i in range(num_timesteps):
            t = torch.full([shape[0]], self.num_timesteps - 1 - i, dtype=torch.long, device=device)
            noise = torch.rand(x.shape + (self.num_classes,), device=device, dtype=torch.float32)
            x, pred_x_start_logits = self.p_sample(model_fn=model_fn, x=x, t=t, noise=noise, edge_features=new_edge_features, edge_index=edge_index, indices=indices)
            if 'num_pred_edges' in self.edge_features:
                new_edge_features[:, -2] = x.flatten().float()
                new_edge_features[:, -1] = torch.sum(x, dim=1).repeat_interleave(self.num_edges) / self.num_edges
            else:
                new_edge_features[:, -1] = x.flatten().float()

        if return_x_init:
            return x_init, x
        else:
            return x    # (val_bs, num_edges)

  # === Loss calculation ===
        
    def cross_entropy_x_start(self, x_start, pred_x_start_logits):
        """Calculate binary weighted cross entropy between x_start and predicted x_start logits.

        Args:
            x_start (torch.Tensor): original clean data, expected binary labels (0 or 1), shape (bs, num_edges)
            pred_x_start_logits (torch.Tensor): logits as predicted by the model

        Returns:
            torch.Tensor: scalar tensor representing the mean binary weighted cross entropy loss.
        """
        # Calculate binary cross-entropy with logits
        x_start = x_start.long().to(self.device, non_blocking=True)
        pred_x_start_logits = pred_x_start_logits.permute(0, 2, 1).float()          # (bs, num_edges, num_classes) -> (bs, num_classes, num_edges)
        pred_x_start_logits = pred_x_start_logits.transpose(1, 2).reshape(-1, 2)    # (bs*num_edges, num_classes)
        x_start = x_start.reshape(-1)                                               # (bs*num_edges)

        ce = F.cross_entropy(pred_x_start_logits, x_start, weight=self.class_weights.float().to(self.device, non_blocking=True), reduction='mean')

        return ce

    def training_losses(self, model_fn, x_start, edge_features, edge_index, indices=None):
        """Training loss calculation."""
        # Add noise to data
        batch_size, num_edges = x_start.shape
        noise = torch.rand(x_start.shape + (self.num_classes,), dtype=torch.float32)
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],))

        # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint itself.
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)  # (bs, num_edges)
        
        # Replace true future with noised future
        x_t = x_t.float()   # (bs, num_edges)
        new_edge_features = edge_features.clone()
        if 'num_pred_edges' in self.edge_features:
            new_edge_features = torch.cat((new_edge_features, torch.zeros((new_edge_features.size(0), 1), device=new_edge_features.device)), dim=1)
        for i in range(x_t.shape[0]):
            if 'num_pred_edges' in self.edge_features:
                new_edge_features[i * num_edges:(i + 1)*num_edges, -2] = x_t[i]
                sum_x_t = torch.sum(x_t[i]).repeat(self.num_edges)
                new_edge_features[i * num_edges:(i + 1)*num_edges, -1] = sum_x_t / self.num_edges
            else:
                new_edge_features[i * num_edges:(i + 1)*num_edges, -1] = x_t[i]
            
        # Calculate the loss
        model_logits, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t, edge_features=new_edge_features, edge_index=edge_index, indices=indices)
        losses = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits)
        
        pred = pred_x_start_logits.argmax(dim=2)    # (batch_size, num_edges, num_classes) -> (batch_size, num_edges)
        
        return losses, pred
    