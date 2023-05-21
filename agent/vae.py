import torch
from torch import nn
from torch.nn import functional as F
from agent.helpers import init_weights


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_size=256) -> None:
        super(VAE, self).__init__()

        self.hidden_size = hidden_size
        self.action_dim = action_dim

        input_dim = state_dim + action_dim

        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                     nn.Mish(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.Mish(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.Mish())

        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_var = nn.Linear(hidden_size, hidden_size)

        self.decoder = nn.Sequential(nn.Linear(hidden_size + state_dim, hidden_size),
                                     nn.Mish(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.Mish(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.Mish())

        self.final_layer = nn.Sequential(nn.Linear(hidden_size, action_dim))

        self.apply(init_weights)

        self.device = device

    def encode(self, action, state):
        x = torch.cat([action, state], dim=-1)
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z, state):
        x = torch.cat([z, state], dim=-1)
        result = self.decoder(x)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss(self, action, state):
        mu, log_var = self.encode(action, state)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z, state)

        kld_weight = 0.1  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, action)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # print('recons_loss: ', recons_loss)
        # print('kld_loss: ', kld_loss)

        loss = recons_loss + kld_weight * kld_loss
        return loss

    def forward(self, state, eval=False):
        batch_size = state.shape[0]
        shape = (batch_size, self.hidden_size)

        if eval:
            z = torch.zeros(shape, device=self.device)
        else:
            z = torch.randn(shape, device=self.device)
        samples = self.decode(z, state)

        return samples.clamp(-1., 1.)
