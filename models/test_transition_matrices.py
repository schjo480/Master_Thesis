import torch
import torch.nn.functional as F
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
        steps = torch.linspace(0, 1, spec['num_timesteps'] + 1, dtype=torch.float64)
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        betas = torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.tensor(0.999))
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
    
    
def _get_gaussian_transition_mat(t):
    r"""Computes transition matrix for q(x_t|x_{t-1}).

    This method constructs a transition matrix Q with
    decaying entries as a function of how far off diagonal the entry is.
    Normalization option 1:
    Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= transition_bands
                1 - \sum_{l \neq i} Q_{il}  if i==j.
                0                          else.

    Normalization option 2:
    tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= transition_bands
                        0                        else.

    Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

    Args:
        t: timestep. integer scalar (or numpy array?)

    Returns:
        Q_t: transition matrix. shape = (num_classes, num_classes).
    """
    num_classes = 2
    device = 'cpu'
    transition_bands = 1
    spec = {'type': 'linear', 'start': 1.0, 'stop': 100.0, 'num_timesteps': 100}
    betas = get_diffusion_betas(spec, device=device)
    transition_bands = transition_bands if transition_bands else num_classes - 1

    beta_t = betas[t]

    mat = torch.zeros((num_classes, num_classes),
                    dtype=torch.float64).to(device, non_blocking=True)

    # Make the values correspond to a similar type of gaussian as in the
    # gaussian diffusion case for continuous state spaces.
    values = torch.linspace(torch.tensor(0.), torch.tensor(num_classes-1), num_classes, dtype=torch.float64).to(device, non_blocking=True)
    values = values * 2./ (num_classes - 1.)
    values = values[:transition_bands+1]
    values = -values * values / beta_t
    
    # To reverse the tensor 'values' starting from the second element
    reversed_values = values[1:].flip(dims=[0])
    # Concatenating the reversed values with the original values
    values = torch.cat([reversed_values, values], dim=0)
    values = F.softmax(values, dim=0)
    values = values[transition_bands:]
    
    for k in range(1, transition_bands + 1):
        off_diag = torch.full((num_classes - k,), values[k], dtype=torch.float64).to(device, non_blocking=True)

        mat += torch.diag(off_diag, k)
        mat += torch.diag(off_diag, -k)

    # Add diagonal values such that rows and columns sum to one.
    # Technically only the ROWS need to sum to one
    # NOTE: this normalization leads to a doubly stochastic matrix,
    # which is necessary if we want to have a uniform stationary distribution.
    diag = 1. - mat.sum(dim=1)
    mat += torch.diag_embed(diag)

    return mat.to(device, non_blocking=True)

for t in range(0, 100, 10):
    print(_get_gaussian_transition_mat(t))

import matplotlib.pyplot as plt

spec = {'type': 'linear', 'start': 1.0, 'stop': 100.0, 'num_timesteps': 100}
betas = get_diffusion_betas(spec, device='cpu')
plt.plot(betas)
plt.show()