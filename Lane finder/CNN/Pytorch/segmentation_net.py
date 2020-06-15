import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
import cv2
import glob
import os
from PIL import Image

# my stuff
from dataloader import image_loader
from model_components import UNet

# for tensorboard and stuff
from torch.utils.tensorboard import SummaryWriter

tb = SummaryWriter(comment='semantic segmentation net')
new_size = (150, 200)

img_files = []
for file in glob.glob(os.path.abspath('C:/Users/turbo/Python projects/Lane-finder/Lane-Finder/data/imgs/*.png')):
    img_files.append(file)

mask_files = []
for file in glob.glob(os.path.abspath('C:/Users/turbo/Python projects/Lane-finder/Lane-Finder/data/masks/*.png')):
    mask_files.append(file)

m = torch.nn.Sigmoid()

def compute_accuracy(Y_target, hypothesis):
    hypothesis = m(hypothesis)
    one_mask = 255*hypothesis.round().cpu().int().numpy()
    tar = 255*Y_target[0].cpu().int().numpy()

    intersection = np.bitwise_and(tar, one_mask).sum()
    union = np.bitwise_or(tar, one_mask).sum()

                                                ## NEEDS FIX ===========
    if intersection == 0 and union == 0:
        pass
    else:
        return intersection/union


def train(model, learning_rate, training_epochs):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    data_loader = image_loader(img_files, mask_files)

    train_cost = []
    train_accu = []
    for epoch in range(training_epochs):
        avg_cost = 0
        avg_iou = []
        for batch_idx, (data, target) in enumerate(data_loader):

            # Select a minibatch
            X = Variable(data.float().cuda())
            Y = Variable(target.float().cuda())

            # initialization of the gradients
            optimizer.zero_grad()

            # Forward propagation: compute the output
            hypothesis = model(X)

            # Computation of the cost J
            cost = criterion(hypothesis, Y)  # <= compute the loss function

            # Backward propagation
            cost.backward()  # <= compute the gradients

            # Update parameters (weights and biais)
            optimizer.step()

            # Print some performance to monitor the training
            train_cost.append(cost.item())
            if batch_idx % 20 == 0:
                print("Epoch= {},\t batch = {},\t cost = {:2.4f},\t accuracy = {}".format(epoch + 1, batch_idx, train_cost[-1],
                                                                                          np.mean(train_accu)))

            # hardcoded batch size
            avg_cost += cost.data / 32
            acc = compute_accuracy(Y, hypothesis)
            if acc is not None:
                avg_iou.append(acc)
        # print("[Epoch: {:>4}], averaged cost = {:>.9}".format(epoch + 1, avg_cost.item()))
        # tb.add_scalar('Loss', avg_cost.item(), epoch)
        # tb.add_scalar('Accuracy', np.mean(avg_iou), epoch)

        # if save_cp:
        #     try:
        #         os.mkdir(dir_checkpoint)
        #     except OSError:
        #         pass
        #     torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                
    tb.close
print('Learning Finished!')


# do you want to save checkpoints?
save_cp = 1
load_state = 1
dir_checkpoint = 'C:\\Users\\turbo\\Python projects\\Lane-finder\\Lane-Finder\\CNN\\\checkpoints\\CONST_NUM_FIL' # file path of checkpoint save

device = torch.device("cuda:0")
net = UNet(3, 1).to(device)

# if load_state:
#     net.load_state_dict(torch.load('C://Users//turbo//Python projects//Lane-finder//Lane-Finder//CNN//checkpoints//CONST_NUM_FILCP_epoch5.pth', map_location=device))

if __name__ == "__main__":
    train(net, 0.001, 10)

