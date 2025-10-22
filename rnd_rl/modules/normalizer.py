import torch 
import torch.nn as nn

# heavily adpoted from the rsl_rl repo.
"""
Observation normalizer and reward normalizer.
"""

class ObsNormalizer(nn.Module):
    """
    EmpericalNormalization in the original repo. Mostly used with observations.
    It normalizes mean and variance of observations using moving average.
    """
    def __init__(
            self,
            obs_dim,
            eps = 1e-2,
            until = None # normalizer will update its values for the first x batches.
        ):
        super(ObsNormalizer, self).__init__()

        # self.device = device
        # this module should be moved to device in ActorCritic or RND class.
        self.eps = eps
        self.until = until

        self.register_buffer("_mean", torch.zeros(obs_dim).unsqueeze(0))
        self.register_buffer("_var", torch.ones(obs_dim).unsqueeze(0))
        self.register_buffer("_std", torch.ones(obs_dim).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values."""

        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        """De-normalize values based on empirical values."""

        return y * (self._std + self.eps) + self._mean



class RewardNormalizer(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize
    the scale of the rewards so that the value function can learn quickly. We did this by dividing
    the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    """

    def __init__(
            self, 
            reward_dim, 
            eps=1e-2, 
            gamma=0.99, 
            until=None):
        super(RewardNormalizer, self).__init__()

        self.emp_norm = ObsNormalizer(reward_dim, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, reward):
        if self.training:
            # update discounted rewards
            avg = self.disc_avg.update(reward)
            # update moments from discounted rewards
            self.emp_norm.update(avg)

        # normalize rewards with the empirical std
        if self.emp_norm._std > 0:
            return reward / self.emp_norm._std
        else:
            return reward


"""
Helper class.
"""


class _DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    Args:
        gamma (float): Discount factor.
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, reward: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = reward
        else:
            self.avg = self.avg * self.gamma + reward
        return self.avg