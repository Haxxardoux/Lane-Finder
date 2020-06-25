

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


latent_tensor = torch.zeros(5,100)
latent_vector = torch.ones(1, 100)

for i in range(5):
    latent_vector = latent_vector*2
    latent_tensor = torch.cat( (latent_tensor[1:], latent_vector), dim=0)
    print(latent_tensor[-2:])
# path = 'C:\\Users\\turbo\\Python projects\\Lane finder\\data\\videos\\test'

import os
import cv2

# path_list = []
# for (dirpath, _, filenames) in os.walk(path):
#     for filename in filenames:
#         path_list.append(os.path.abspath(os.path.join(path, filename)))

# for filename in path_list:
#     vcap = cv2.VideoCapture(filename)
#     if vcap.isOpened():
#         # get vcap property
#         width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         print(width)
#         print(height)

