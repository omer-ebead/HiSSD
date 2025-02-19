from net_utils import TruncatedNormal
import torch
import torch.nn as nn

a = torch.rand(1, 4).requires_grad_(True)
b = torch.rand(1, 4).requires_grad_(True)

fc = nn.Linear(4, 5)

dist = TruncatedNormal(a, b)
c = dist.sample(clip=None)
d = fc(c)
print(a)
print(b)
print(c)
print(d)

label = torch.rand(1, 5)
loss = (label - d) ** 2
loss.backward()