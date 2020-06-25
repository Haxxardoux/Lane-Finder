

import torch


one = torch.ones((2, 10))/10
two = torch.ones(2, 10)/10
# two = torch.Tensor([[2, 2, 2, 2], [1, 1, 1, 1], [3, 3, 3, 3]])

eye = torch.eye(len(two), dtype = torch.bool)

thing1 = torch.matmul(two, two.T)*~eye
thing2 = torch.sum(torch.matmul(two, two.T)*~eye, dim=1).unsqueeze(dim=1)

foo = torch.ger(torch.rand(5).float(), torch.ones(100))

test1 = 10*torch.ones(3, 5)+torch.rand(3, 5)

test2 = 11*torch.ones(3, 5)+torch.rand(3, 5)

doikt = {'Thing':1, 'Thing2':2}
for key, value in doikt.items():
    print('its dat boi! ', key)
    print('my neck ', value)
