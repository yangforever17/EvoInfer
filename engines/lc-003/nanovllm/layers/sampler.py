import torch
from torch import nn


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy = logits.argmax(dim=-1)
        if bool((temperatures == 0).all()):
            return greedy
        t = torch.where(temperatures == 0, torch.ones_like(temperatures), temperatures)
        probs = torch.softmax(logits.float().div_(t.unsqueeze(dim=1)), dim=-1)
        sampled = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy, sampled)
