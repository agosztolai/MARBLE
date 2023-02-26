"""
MAIN FILE

Definition of network classes, training functionality.
"""

import sys
sys.path.append("./RNN_scripts")
from helpers import gram_schmidt_pt
import torch.nn as nn
from math import sqrt, floor
import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # Compute loss for each (trial, timestep) (average accross output dimensions)
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    # Account for different number of masked values per trial
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    return loss_by_trial.mean()


def train(net, _input, _target, _mask, n_epochs, random_ic = False, lr=1e-2, batch_size=32, plot_learning_curve=False, plot_gradient=False,
          mask_gradients=False, clip_gradient=None, early_stop=None, keep_best=False, cuda=False, resample=False):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param mask_gradients: bool, set to True if training the SupportLowRankRNN_withMask for reduced models
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param early_stop: None or float, set to target loss value after which to immediately stop if attained
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :param resample: for SupportLowRankRNNs, set True
    :return: nothing
    """
    print("Training...")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    num_examples = _input.shape[0]
    all_losses = []
    if plot_gradient:
        gradient_norms = []

    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    # Initialize setup to keep best network
    with torch.no_grad():
        output, _ = net(input)
        initial_loss = loss_mse(output, target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
        if keep_best:
            best = net.clone()
            best_loss = initial_loss.item()
            
    if random_ic:
        print('training with random initial conditions')

    # Training loop
    for epoch in range(n_epochs):
        begin = time.time()
        losses = []  # losses over the whole epoch
        for i in range(num_examples // batch_size):
            optimizer.zero_grad()
            random_batch_idx = random.sample(range(num_examples), batch_size)
            batch = input[random_batch_idx]
            
            #set initial condition
            if random_ic:
                # norm = net.m.norm(dim=0)
                h0 = torch.rand(net.hidden_size).to(device)*net.hidden_size/300
                net.h0.data = h0
            
            output, _ = net(batch)
            loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
            losses.append(loss.item())
            all_losses.append(loss.item())
            loss.backward()
            if mask_gradients:
                net.m.grad = net.m.grad * net.m_mask
                net.n.grad = net.n.grad * net.n_mask
                net.wi.grad = net.wi.grad * net.wi_mask
                net.wo.grad = net.wo.grad * net.wo_mask
                net.unitn.grad = net.unitn.grad * net.unitn_mask
                net.unitm.grad = net.unitm.grad * net.unitm_mask
                net.unitwi.grad = net.unitwi.grad * net.unitwi_mask
            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
            if plot_gradient:
                tot = 0
                for param in [p for p in net.parameters() if p.requires_grad]:
                    tot += (param.grad ** 2).sum()
                gradient_norms.append(sqrt(tot))
            optimizer.step()
            # These 2 lines important to prevent memory leaks
            loss.detach_()
            output.detach_()
            if resample:
                net.resample_basis()
        if keep_best and np.mean(losses) < best_loss:
            best = net.clone()
            best_loss = np.mean(losses)
            print("epoch %d:  loss=%.3f  (took %.2f s) *" % (epoch, np.mean(losses), time.time() - begin))
        else:
            print("epoch %d:  loss=%.3f  (took %.2f s)" % (epoch, np.mean(losses), time.time() - begin))
        if early_stop is not None and np.mean(losses) < early_stop:
            break

    if plot_learning_curve:
        plt.plot(all_losses)
        plt.title("Learning curve")
        plt.show()

    if plot_gradient:
        plt.plot(gradient_norms)
        plt.title("Gradient norm")
        plt.show()

    if keep_best:
        net.load_state_dict(best.state_dict())


class FullRankRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha=0.2, rho=1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False,
                 wi_init=None, wo_init=None, wrec_init=None, si_init=None, so_init=None):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float, value of dt/tau
        :param rho: float, std of gaussian distribution for initialization
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param si_init: input scaling, torch tensor of shape (input_dim)
        :param so_init: output scaling, torch tensor of shape (output_dim)
        """
        super(FullRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.non_linearity = torch.tanh

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if wrec_init is None:
                self.wrec.normal_(std=rho / sqrt(hidden_size))
            else:
                self.wrec.copy_(wrec_init)
            if wo_init is None:
                self.wo.normal_(std=1 / hidden_size)
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            self.h0.zero_()
        self.wi_full, self.wo_full = [None] * 2
        self._define_proxy_parameters()

    def _define_proxy_parameters(self):
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so

    def forward(self, input):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :return: (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(h)
        self._define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.wrec.device)

        # simulation loop
        for i in range(seq_len):
            h = h + \
                self.noise_std * noise[:, i, :] + \
                self.alpha * (-h + r.matmul(self.wrec.t()) + \
                input[:, i, :].matmul(self.wi_full))
                    
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full)
            trajectories[:, i, :] = h

        return output, trajectories

    def clone(self):
        new_net = FullRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                              self.rho, self.train_wi, self.train_wo, self.train_wrec, self.train_h0,
                              self.wi, self.wo, self.wrec, self.si, self.so)
        return new_net
    
    
def simulation_loop(model, input):
    batch_size = input.shape[0]
    seq_len = input.shape[1]
    h = model.h0
    r = model.non_linearity(h)
    
    noise = torch.randn(batch_size, seq_len, model.hidden_size, device=model.m.device)
    output = torch.zeros(batch_size, seq_len, model.output_size, device=model.m.device)
    trajectories = torch.zeros(batch_size, seq_len, model.hidden_size, device=model.m.device)
    
    for i in range(seq_len):
        h = h + \
            model.noise_std * noise[:, i, :] + \
            model.alpha * (-h + r.matmul(model.n).matmul(model.m.t()) / model.hidden_size + \
            input[:, i, :].matmul(model.wi_full))
                
        r = model.non_linearity(h)
        output[:, i, :] = r.matmul(model.wo_full) / model.hidden_size
        trajectories[:, i, :] = h
        
    return output, trajectories


class LowRankRNN(nn.Module):
    """
    This class implements the low-rank RNN. Instead of being parametrized by an NxN connectivity matrix, it is
    parametrized by two Nxr matrices m and n such that the connectivity is m * n^T
    """

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rank=1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False, train_si=True, train_so=True,
                 wi_init=None, wo_init=None, m_init=None, n_init=None, si_init=None, so_init=None, h0_init=None):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float, value of dt/tau
        :param rank: int
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param train_si: bool
        :param train_so: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param m_init: torch tensor of shape (hidden_size, rank)
        :param n_init: torch tensor of shape (hidden_size, rank)
        :param si_init: torch tensor of shape (input_size)
        :param so_init: torch tensor of shape (output_size)
        :param h0_init: torch tensor of shape (hidden_size)
        """
        super(LowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rank = rank
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.train_si = train_si
        self.train_so = train_so
        self.non_linearity = torch.tanh

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        if not train_si:
            self.si.requires_grad = False
        self.m = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.n = nn.Parameter(torch.Tensor(hidden_size, rank))
        if not train_wrec:
            self.m.requires_grad = False
            self.n.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
        if not train_so:
            self.so.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if m_init is None:
                self.m.normal_()
            else:
                self.m.copy_(m_init)
            if n_init is None:
                self.n.normal_()
            else:
                self.n.copy_(n_init)
            if wo_init is None:
                self.wo.normal_(std=4.)
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            if h0_init is None:
                self.h0.zero_()
            else:
                self.h0.copy_(h0_init)
        self.wrec, self.wi_full, self.wo_full = [None] * 3
        self._define_proxy_parameters()

    def _define_proxy_parameters(self):
        self.wrec = None
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so

    def forward(self, input):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :return: (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        return simulation_loop(self, input)

    def clone(self):
        new_net = LowRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                             self.rank, self.train_wi, self.train_wo, self.train_wrec, self.train_h0, self.train_si,
                             self.train_so, self.wi, self.wo, self.m, self.n, self.si, self.so)
        new_net._define_proxy_parameters()
        return new_net

    def load_state_dict(self, state_dict, strict=True):
        """
        override
        """
        if 'rec_noise' in state_dict:
            del state_dict['rec_noise']
        super().load_state_dict(state_dict, strict)
        self._define_proxy_parameters()

    def svd_reparametrization(self):
        """
        Orthogonalize m and n via SVD
        """
        with torch.no_grad():
            structure = (self.m @ self.n.t()).numpy()
            m, s, n = np.linalg.svd(structure, full_matrices=False)
            m, s, n = m[:, :self.rank], s[:self.rank], n[:self.rank, :]
            self.m.set_(torch.from_numpy(m * np.sqrt(s)))
            self.n.set_(torch.from_numpy(n.transpose() * np.sqrt(s)))
            self._define_proxy_parameters()


class SupportLowRankRNN(nn.Module):
    """
    This class implements the mixture-of-gaussians, low-rank RNN. The difference with the low-rank RNN is that
    all vectors are defined as transformation of a gaussian basis of dimensionality b for each population.

    For example the matrix m, instead of having Nxr free parameters, is parametrized by a tensor
    m_weights of shape (r, p, b) (where r is the rank, p is the number of populations). A gaussian basis of
    shape Nxb is sampled, and m is then computed from the basis and the weights, by assigning each neuron to a
    population monotonically.

    The weights defined above correspond to a linear transformation of the gaussian basis (ie the expectancy
    of the final distribution obtained is always zero). Affine transforms can be defined by setting biases.
    """

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rank=1, n_supports=1, weights=None,
                 gaussian_basis_dim=None, m_weights_init=None, n_weights_init=None, wi_weights_init=None,
                 wo_weights_init=None, m_biases_init=None, n_biases_init=None, wi_biases_init=None, train_biases=False):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float
        :param rank: int
        :param n_supports: int, number of cell classes used
        :param weights: list, proportion of total population for each cell class (GMM components weights)
        :param gaussian_basis_dim: dimensionality of the gaussian basis on which weights are learned
        :param m_weights_init: torch tensor of shape (rank, n_supports, gaussian_basis_dim)
        :param n_weights_init: torch tensor of shape (rank, n_supports, gaussian_basis_dim)
        :param wi_weights_init: torch tensor of shape (input_size, n_supports, self.gaussian_basis_dim)
        :param wo_weights_init: torch tensor of shape (output_size, n_supports, self.gaussian_basis_dim)
        :param m_biases_init: torch tensor of shape (rank, n_supports)
        :param n_biases_init: torch tensor of shape (rank, n_supports)
        :param wi_biases_init: torch tensor of shape (input_size, n_supports)
        :param train_biases: bool
        """
        super(SupportLowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rank = rank
        self.n_supports = n_supports
        self.gaussian_basis_dim = 2 * rank + input_size if gaussian_basis_dim is None else gaussian_basis_dim
        self.non_linearity = torch.tanh

        self.gaussian_basis = nn.Parameter(torch.randn((self.gaussian_basis_dim, hidden_size)), requires_grad=False)
        self.supports = nn.Parameter(torch.zeros((n_supports, hidden_size)), requires_grad=False)
        if weights is None:
            self.weights = nn.Parameter(torch.tensor([1 / hidden_size]))
            l_support = hidden_size // n_supports
            for i in range(n_supports):
                self.supports[i, l_support * i: l_support * (i + 1)] = 1
            self.weights = [l_support / hidden_size] * n_supports
        else:
            k = 0
            self.weights = nn.Parameter(torch.tensor(weights), requires_grad=False)
            for i in range(n_supports):
                self.supports[i, k: k + floor(weights[i] * hidden_size)] = 1
                k += floor(weights[i] * hidden_size)

        # Define parameters
        self.wi_weights = nn.Parameter(torch.Tensor(input_size, n_supports, self.gaussian_basis_dim))
        self.m_weights = nn.Parameter(torch.Tensor(rank, n_supports, self.gaussian_basis_dim))
        self.n_weights = nn.Parameter(torch.Tensor(rank, n_supports, self.gaussian_basis_dim))
        self.wo_weights = nn.Parameter(torch.Tensor(output_size, n_supports, self.gaussian_basis_dim))
        self.wi_biases = nn.Parameter(torch.Tensor(input_size, n_supports), requires_grad=train_biases)
        self.m_biases = nn.Parameter(torch.Tensor(rank, n_supports), requires_grad=train_biases)
        self.n_biases = nn.Parameter(torch.Tensor(rank, n_supports), requires_grad=train_biases)
        self.h0_weights = nn.Parameter(torch.Tensor(n_supports, self.gaussian_basis_dim))
        self.h0_weights.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_weights_init is not None:
                self.wi_weights.copy_(wi_weights_init)
            else:
                self.wi_weights.normal_()
            if m_weights_init is not None:
                self.m_weights.copy_(m_weights_init)
            else:
                self.m_weights.normal_(std=1 / sqrt(hidden_size))
            if n_weights_init is not None:
                self.n_weights.copy_(n_weights_init)
            else:
                self.n_weights.normal_(std=1 / sqrt(hidden_size))
            if wo_weights_init is not None:
                self.wo_weights.copy_(wo_weights_init)
            else:
                self.wo_weights.normal_(std=1 / hidden_size)
            if wi_biases_init is not None:
                self.wi_biases.copy_(wi_biases_init)
            else:
                self.wi_biases.zero_()
            if m_biases_init is not None:
                self.m_biases.copy_(m_biases_init)
            else:
                self.m_biases.zero_()
            if n_biases_init is not None:
                self.n_biases.copy_(n_biases_init)
            else:
                self.n_biases.zero_()
            self.h0_weights.zero_()
        self.wi, self.m, self.n, self.wo, self.h0, self.wi_full, self.wo_full = [None] * 7
        self._define_proxy_parameters()

    def _define_proxy_parameters(self):
        self.wi = torch.sum((self.wi_weights @ self.gaussian_basis) * self.supports, dim=(1,)) + \
                  self.wi_biases @ self.supports
        self.wi_full = self.wi
        self.m = torch.sum((self.m_weights @ self.gaussian_basis) * self.supports, dim=(1,)).t() + \
                 (self.m_biases @ self.supports).t()
        self.n = torch.sum((self.n_weights @ self.gaussian_basis) * self.supports, dim=(1,)).t() + \
                 (self.n_biases @ self.supports).t()
        self.wo = torch.sum((self.wo_weights @ self.gaussian_basis) * self.supports, dim=(1,)).t()
        self.wo_full = self.wo
        self.h0 = torch.sum((self.h0_weights @ self.gaussian_basis) * self.supports, dim=(0,))

    def forward(self, input):
        return simulation_loop(self, input)

    def clone(self):
        new_net = SupportLowRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                                    self.rank, self.n_supports, self.weights.tolist(), self.gaussian_basis_dim,
                                    self.m_weights, self.n_weights, self.wi_weights, self.wo_weights, self.m_biases,
                                    self.n_biases, self.wi_biases)
        new_net.gaussian_basis.copy_(self.gaussian_basis)
        new_net._define_proxy_parameters()
        return new_net

    def load_state_dict(self, state_dict, strict=True):
        """
        override to recompute w_rec on loading
        """
        super().load_state_dict(state_dict, strict)
        self._define_proxy_parameters()

    def resample_basis(self):
        self.gaussian_basis.normal_()
        self._define_proxy_parameters()


class SupportLowRankRNN_withMask(nn.Module):
    """
    This network has been defined to train an arbitrary subset of the parameters offered by the SupportLowRankRNN
    by adding a mask.
    """

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rank=1, n_supports=1,
                 gaussian_basis_dim=None, initial_m=None, initial_n=None, initial_unitm=None, initial_unitn=None,
                 initial_wi=None, initial_unitwi=None, initial_wo=None,
                 initial_h0=None, initial_unith0=None, initial_bias=None, train_h0=False, train_bias=False,
                 initial_wi_mask=None, initial_wo_mask=None, initial_m_mask=None, initial_n_mask=None):
        super(SupportLowRankRNN_withMask, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rank = rank
        self.n_supports = n_supports
        self.gaussian_basis_dim = 2 * rank + input_size if gaussian_basis_dim is None else gaussian_basis_dim
        self.non_linearity = torch.tanh

        self.gaussian_basis = nn.Parameter(torch.randn((self.gaussian_basis_dim, hidden_size)), requires_grad=False)
        self.unit_vector = nn.Parameter(torch.ones((1, hidden_size)), requires_grad=False)
        self.supports = nn.Parameter(torch.zeros((n_supports, hidden_size)), requires_grad=False)
        l_support = hidden_size // n_supports
        for i in range(n_supports):
            self.supports[i, l_support * i: l_support * (i + 1)] = 1

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, n_supports, self.gaussian_basis_dim))
        self.unitwi = nn.Parameter(torch.Tensor(input_size, n_supports, 1))
        self.m = nn.Parameter(torch.Tensor(rank, n_supports, self.gaussian_basis_dim))
        self.n = nn.Parameter(torch.Tensor(rank, n_supports, self.gaussian_basis_dim))
        self.unitm = nn.Parameter(torch.Tensor(rank, n_supports, 1))
        self.unitn = nn.Parameter(torch.Tensor(rank, n_supports, 1))
        self.wo = nn.Parameter(torch.Tensor(output_size, n_supports, self.gaussian_basis_dim))
        self.h0 = nn.Parameter(torch.Tensor(n_supports, self.gaussian_basis_dim))
        self.unith0 = nn.Parameter(torch.Tensor(n_supports, 1))
        self.bias = nn.Parameter(torch.Tensor(n_supports, 1))

        self.wi_mask = nn.Parameter(torch.Tensor(input_size, n_supports, self.gaussian_basis_dim), requires_grad=False)
        self.unitwi_mask = nn.Parameter(torch.Tensor(input_size, n_supports, 1), requires_grad=False)
        self.m_mask = nn.Parameter(torch.Tensor(rank, n_supports, self.gaussian_basis_dim), requires_grad=False)
        self.n_mask = nn.Parameter(torch.Tensor(rank, n_supports, self.gaussian_basis_dim), requires_grad=False)
        self.unitm_mask = nn.Parameter(torch.Tensor(rank, n_supports, 1), requires_grad=False)
        self.unitn_mask = nn.Parameter(torch.Tensor(rank, n_supports, 1), requires_grad=False)
        self.wo_mask = nn.Parameter(torch.Tensor(output_size, n_supports, self.gaussian_basis_dim), requires_grad=False)
        self.h0_mask = nn.Parameter(torch.Tensor(n_supports, self.gaussian_basis_dim), requires_grad=False)
        self.unith0_mask = nn.Parameter(torch.Tensor(n_supports, 1), requires_grad=False)
        self.bias_mask = nn.Parameter(torch.Tensor(n_supports, 1), requires_grad=False)

        if not train_h0:
            self.h0.requires_grad = False
            self.unith0.requires_grad = False
        if not train_bias:
            self.bias.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if initial_wi is not None:
                self.wi.copy_(initial_wi)
                if initial_wi_mask is not None:
                    maskc = initial_wi_mask
                else:
                    maskc = torch.where(initial_wi != 0, torch.ones_like(initial_wi), torch.zeros_like(initial_wi))
                self.wi_mask.copy_(maskc)
            else:
                self.wi.zero_()
                self.wi_mask.zero_()
            if initial_unitwi is not None:
                self.unitwi.copy_(initial_unitwi)
                maskc = torch.where(initial_unitwi != 0, torch.ones_like(initial_unitwi),
                                    torch.zeros_like(initial_unitwi))
                self.unitwi_mask.copy_(maskc)
            else:
                self.unitwi.zero_()
                self.unitwi_mask.zero_()
            if initial_m is not None:
                self.m.copy_(initial_m)
                if initial_m_mask is not None:
                    maskc = initial_m_mask
                else:
                    maskc = torch.where(initial_m != 0, torch.ones_like(initial_m), torch.zeros_like(initial_m))
                self.m_mask.copy_(maskc)
            else:
                self.m.zero_()
                self.m_mask.zero_()
            if initial_n is not None:
                self.n.copy_(initial_n)
                if initial_n_mask is not None:
                    maskc = initial_n_mask
                else:
                    maskc = torch.where(initial_n != 0, torch.ones_like(initial_n), torch.zeros_like(initial_n))
                self.n_mask.copy_(maskc)
            else:
                self.n.zero_()
                self.n_mask.zero_()
            if initial_unitm is not None:
                self.unitm.copy_(initial_unitm)
                maskc = torch.where(initial_unitm != 0, torch.ones_like(initial_unitm), torch.zeros_like(initial_unitm))
                self.unitm_mask.copy_(maskc)
            else:
                self.unitm.zero_()
                self.unitm_mask.zero_()
            if initial_unitn is not None:
                self.unitn.copy_(initial_unitn)
                maskc = torch.where(initial_unitn != 0, torch.ones_like(initial_unitn), torch.zeros_like(initial_unitn))
                self.unitn_mask.copy_(maskc)
            else:
                self.unitn.zero_()
                self.unitn_mask.zero_()
            if initial_wo is not None:
                self.wo.copy_(initial_wo)
                if initial_wo_mask is not None:
                    maskc = initial_wo_mask
                else:
                    maskc = torch.where(initial_wo != 0, torch.ones_like(initial_wo), torch.zeros_like(initial_wo))
                self.wo_mask.copy_(maskc)
            else:
                self.wo.zero_()
                self.wo_mask.zero_()
            if initial_h0 is not None:
                self.h0.copy_(initial_h0)
                maskc = torch.where(initial_h0 != 0, torch.ones_like(initial_h0), torch.zeros_like(initial_h0))
                self.h0_mask.copy_(maskc)
            else:
                self.h0.zero_()
                self.h0_mask.zero_()
            if initial_unith0 is not None:
                self.unith0.copy_(initial_unith0)
                maskc = torch.where(initial_unith0 != 0, torch.ones_like(initial_unith0),
                                    torch.zeros_like(initial_unith0))
                self.unith0_mask.copy_(maskc)
            else:
                self.unith0.zero_()
                self.unith0_mask.zero_()
            if initial_bias is not None:
                self.bias.copy_(initial_bias)
                maskc = torch.where(initial_bias != 0, torch.ones_like(initial_bias), torch.zeros_like(initial_bias))
                self.bias_mask.copy_(maskc)
            else:
                self.bias.zero_()
                self.bias_mask.zero_()

        self.wi_full, self.m_rec, self.n_rec, self.wo_full, self.w_rec, self.h0_full, self.bias_full = [None] * 7
        self.define_proxy_parameters()

    def define_proxy_parameters(self):
        self.wi_full = torch.sum((self.wi @ self.gaussian_basis) * self.supports, dim=(1,)) + \
                       torch.sum((self.unitwi @ self.unit_vector) * self.supports, dim=(1,))
        self.m_rec = torch.sum((self.m @ self.gaussian_basis) * self.supports, dim=(1,)).t() + \
                     torch.sum((self.unitm @ self.unit_vector) * self.supports, dim=(1,)).t()
        self.n_rec = torch.sum((self.n @ self.gaussian_basis) * self.supports, dim=(1,)).t() + \
                     torch.sum((self.unitn @ self.unit_vector) * self.supports, dim=(1,)).t()
        self.wo_full = torch.sum((self.wo @ self.gaussian_basis) * self.supports, dim=(1,)).t()
        self.w_rec = self.m_rec.matmul(self.n_rec.t())
        self.h0_full = torch.sum((self.h0 @ self.gaussian_basis) * self.supports, dim=(0,)) + \
                       torch.sum((self.unith0 @ self.unit_vector) * self.supports, dim=(0,))
        self.bias_full = torch.sum((self.bias @ self.unit_vector) * self.supports, dim=(0,))

    def forward(self, input):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        self.define_proxy_parameters()
        h = self.h0_full
        r = self.non_linearity(h)
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.m_rec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.m_rec.device)
        trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.m_rec.device)

        # simulation loop
        for i in range(seq_len):
            h = h + \
                self.bias_full + \
                self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(self.w_rec.t()) + \
                input[:, i, :].matmul(self.wi_full))
                    
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full)
            trajectories[:, i, :] = h

        return output, trajectories

    def clone(self):
        new_net = SupportLowRankRNN_withMask(self.input_size, self.hidden_size, self.output_size, self.noise_std,
                                             self.alpha,
                                             self.rank, self.n_supports, self.gaussian_basis_dim, self.m, self.n,
                                             self.unitm, self.unitn, self.wi,
                                             self.unitwi, self.wo, self.h0, self.unith0, self.bias)
        new_net.gaussian_basis.copy_(self.gaussian_basis)
        new_net.define_proxy_parameters()
        return new_net

    def load_state_dict(self, state_dict, strict=True):
        """
        override to recompute w_rec on loading
        """
        super().load_state_dict(state_dict, strict)
        self.define_proxy_parameters()

    def resample_basis(self):
        self.gaussian_basis.normal_()
        self.define_proxy_parameters()

    def orthogonalize_basis(self):
        for i in range(self.n_supports):
            gaussian_chunk = self.gaussian_basis[:, self.supports[i] == 1].view(self.gaussian_basis_dim, -1)
            gram_schmidt_pt(gaussian_chunk)
            self.gaussian_basis[:, self.supports[i] == 1] = gaussian_chunk
        self.gaussian_basis *= sqrt(self.hidden_size // self.n_supports)
        self.define_proxy_parameters()
        
        
def simulation_loop(model, input):
    batch_size = input.shape[0]
    seq_len = input.shape[1]
    h = model.h0
    r = model.non_linearity(h)
    
    noise = torch.randn(batch_size, seq_len, model.hidden_size, device=model.m.device)
    output = torch.zeros(batch_size, seq_len, model.output_size, device=model.m.device)
    trajectories = torch.zeros(batch_size, seq_len, model.hidden_size, device=model.m.device)
    
    for i in range(seq_len):
        h = h + \
            model.noise_std * noise[:, i, :] + \
            model.alpha * (-h + r.matmul(model.n).matmul(model.m.t()) / model.hidden_size + \
            input[:, i, :].matmul(model.wi_full))
                
        r = model.non_linearity(h)
        output[:, i, :] = r.matmul(model.wo_full) / model.hidden_size
        trajectories[:, i, :] = h
        
    return output, trajectories


class OptimizedLowRankRNN(nn.Module):
    """
    LowRankRNN class with a different definition of scalings (see caption of SI Fig. about the 3-population Ctx net)
    """

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rho=0., rank=1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False, train_si=True, train_so=True,
                 wi_init=None, wo_init=None, m_init=None, n_init=None, si_init=None, so_init=None, h0_init=None):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float
        :param rho: float, std of quenched noise matrix
        :param rank: int
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param train_si: bool (can't be True if train_wi is already True)
        :param train_so: bool (can't be True if train_wo is already True)
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param m_init: torch tensor of shape (hidden_size, rank)
        :param n_init: torch tensor of shape (hidden_size, rank)
        :param si_init: input scaling, torch tensor of shape (input_dim)
        :param so_init: output scaling, torch tensor of shape (output_dim)
        :param h0_init: torch tensor of shape (hidden_size)
        """
        super(OptimizedLowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.rank = rank
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.train_si = train_si
        self.train_so = train_so
        self.non_linearity = torch.tanh

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        if not train_si:
            self.si.requires_grad = False
        self.m = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.n = nn.Parameter(torch.Tensor(hidden_size, rank))
        if not train_wrec:
            self.m.requires_grad = False
            self.n.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        else:
            self.wo.requires_grad = False
        if not train_so:
            self.so.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if m_init is None:
                self.m.normal_(std=1 / sqrt(hidden_size))
            else:
                self.m.copy_(m_init)
            if n_init is None:
                self.n.normal_(std=1 / sqrt(hidden_size))
            else:
                self.n.copy_(n_init)
            if wo_init is None:
                self.wo.normal_(std=2 / hidden_size)
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            if h0_init is None:
                self.h0.zero_()
            else:
                self.h0.copy_(h0_init)
        self.wrec, self.wi_full, self.wo_full = [None] * 3
        self.define_proxy_parameters()

    def define_proxy_parameters(self):
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so

    def forward(self, input):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :return: (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(h)
        self.define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.m.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.m.device)
        trajectories = torch.zeros(batch_size, seq_len+1, self.hidden_size, device=self.m.device)
        trajectories[:, 0, :] = h

        # simulation loop
        for i in range(seq_len):
            h = h + \
                self.noise_std * noise[:, i, :] + \
                self.alpha * (-h + r.matmul(self.n).matmul(self.m.t()) + \
                input[:, i, :].matmul(self.wi_full))
                    
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full)
            trajectories[:, i+1, :] = h

        return output, trajectories

    def clone(self):
        new_net = OptimizedLowRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                             self.rho, self.rank, self.train_wi, self.train_wo, self.train_wrec, self.train_h0,
                             self.train_si, self.train_so, self.wi, self.wo, self.m, self.n, self.si, self.so)
        new_net.define_proxy_parameters()
        return new_net

    def resample_connectivity_noise(self):
        self.define_proxy_parameters()

    def load_state_dict(self, state_dict, strict=True):
        """
        override to recompute w_rec on loading
        """
        super().load_state_dict(state_dict, strict)
        self.define_proxy_parameters()

    def svd_reparametrization(self):
        """
        Orthogonalize m and n via SVD
        """
        with torch.no_grad():
            structure = (self.m @ self.n.t()).numpy()
            m, s, n = np.linalg.svd(structure, full_matrices=False)
            m, s, n = m[:, :self.rank], s[:self.rank], n[:self.rank, :]
            self.m.set_(torch.from_numpy(m * np.sqrt(s)))
            self.n.set_(torch.from_numpy(n.transpose() * np.sqrt(s)))
            self.define_proxy_parameters()