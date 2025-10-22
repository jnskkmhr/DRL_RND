import torch 

class TrajData:
    def __init__(self, n_steps, n_envs, n_obs, n_actions):
        s, e, o, a = n_steps, n_envs, n_obs, n_actions

        self.states = torch.zeros((s, e, o))
        self.actions = torch.zeros((s, e, a))
        self.rewards = torch.zeros((s, e))
        self.extrinsic_rewards = torch.zeros((s, e))
        self.not_dones = torch.zeros((s, e))

        self.log_probs = torch.zeros((s, e))
        self.returns = torch.zeros((s, e))
        self.advantages = torch.zeros((s, e))
        self.values = torch.zeros((s, e))
        self.n_steps = s

    def detach(self):
        self.actions = self.actions.detach()
        self.log_probs = self.log_probs.detach()

    def store(self, t, s, a, r, lp, d):
        self.states[t] = s
        self.actions[t] = a
        self.rewards[t] = torch.Tensor(r)
        self.extrinsic_rewards[t] = torch.Tensor(r)
        self.log_probs[t] = lp
        self.not_dones[t] = 1 - torch.Tensor(d)

    def calc_returns(self, values, last_value, gamma = .99, gae_lambda = 0.999 ):
        self.returns = self.rewards.clone()
        self.values = values.clone()
        last_value = last_value.squeeze()
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps-1:
                delta = self.rewards[t] + gamma * last_value * self.not_dones[t] - self.values[t]
                self.advantages[t] = delta
            else:
                delta = self.rewards[t] + gamma * self.values[t+1] * self.not_dones[t] - self.values[t]
                self.advantages[t] = delta + gamma * gae_lambda * self.not_dones[t] * self.advantages[t+1]
            self.returns[t] = self.advantages[t] + self.values[t]
