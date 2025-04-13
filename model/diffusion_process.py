'''
TODO
[] beta schedule
[] forward process
[] u-net modification
[] loss function
[] training loop
[] sampling
'''
import torch
# Linear Noise Scheduler used for both forward and reverse process
# given xT it gives x(T-1)
class NoiseScheduler:
    def __init__(self, time_steps, beta_start, beta_finish):
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_finish = beta_finish

        # beta increases over time
        self.all_beta = torch.linspace(beta_start,beta_finish,time_steps)
        self.all_alpha = 1-self.all_beta
        self.alpha_cum_prod = torch.cumprod(self.all_alpha, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1-self.alpha_cum_prod)

    # for forward process add noise (in math equation x0 - initial_sample, epsilon - noise)
    def forward_add_noise(self, initial_sample, noise, timestep):
        batch_size = initial_sample.shape[0]
        # [batch_size, channels, height, width]
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[timestep].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[timestep].reshape(batch_size)

        for _ in range(len(batch_size)-1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # diffusion formula
        return sqrt_alpha_cum_prod*initial_sample + sqrt_one_minus_alpha_cum_prod*noise

    # math formula as I wrote it, chat suggested more professonal below:
    '''
    def forward_add_noise(self, initial_sample, noise, timestep):
    # initial_sample: [B, C, H, W]
    batch_size = initial_sample.shape[0]

    sqrt_alpha_cumprod = self.sqrt_alpha_cum_prod[timestep].view(batch_size, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cum_prod[timestep].view(batch_size, 1, 1, 1)

    return sqrt_alpha_cumprod * initial_sample + sqrt_one_minus_alpha_cumprod * noise
    '''

    # sample x_{t-1} given x_t and the predicted noise at specific timestep
    def reverse_diffusion_step(self, x_t, predicted_noise, timestep):
        x_0_pred = (x_t - self.sqrt_one_minus_alpha_cum_prod[timestep] * predicted_noise) \
                   / self.sqrt_alpha_cum_prod[timestep]
        x_0_pred = torch.clamp(x_0_pred, -1., 1.)

        # mean of the posterior q(x_{t-1} | x_t, x_0)
        posterior_mean = x_t - (self.all_beta[timestep] * predicted_noise) \
                         / self.sqrt_one_minus_alpha_cum_prod[timestep]
        posterior_mean = posterior_mean / self.all_alpha[timestep].sqrt()

        if timestep == 0:
            return posterior_mean, x_0_pred
        # variance and standard deviation of the posterior
        posterior_variance = (1 - self.alpha_cum_prod[timestep - 1]) \
                             / (1 - self.alpha_cum_prod[timestep])
        posterior_variance *= self.all_beta[timestep]
        std_dev = posterior_variance.sqrt()

        noise = torch.randn_like(x_t)
        x_prev = posterior_mean + std_dev * noise

        return x_prev, x_0_pred

