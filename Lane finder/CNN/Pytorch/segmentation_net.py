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
from dataloader import train_image_loader, val_image_loader
from model_components import UNet
from utils import Params

# for logging experiments
import mlflow

# set the file path where the logs will be stored. this should be a global reference since many different scripts will reference it from different directories
mlflow.tracking.set_tracking_uri('file:\\Users\\turbo\\Python projects\\Lane finder\\Logs')

# a new experiment will be created if one by that name does not already exists
mlflow.set_experiment('Weird U-Net')


new_size = (150, 200)

img_files, mask_files = [], []
for file in glob.glob(os.path.abspath('C:/Users/turbo/Python projects/Lane finder/data/imgs/*.png')):
    img_files.append(file)
for file in glob.glob(os.path.abspath('C:/Users/turbo/Python projects/Lane finder/data/masks/*.png')):
    mask_files.append(file)

m = torch.nn.Sigmoid()

def compute_accuracy(Y_target, hypothesis):
    hypothesis = m(hypothesis)
    one_mask = hypothesis.round()
    tar = Y_target[0]

    intersection = (tar.int() & one_mask.int()).sum()
    union = (tar*one_mask).sum()

    if not (intersection == 0) & (union == 0):
        return intersection/union


def train(model, train_data, val_data, learning_rate, training_epochs, optimizer):
    # this will be used no matter what in the case of u-net
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(training_epochs):
        train_loss, train_iou = 0, []
        val_loss, val_iou = 0, []

        for batch_idx, (data, target) in enumerate(train_data):

            # Select a minibatch
            X = data.to(device)
            Y = target.to(device).float()

            # set parameter gradients to zero 
            optimizer.zero_grad()

            # Forward pass: compute the output
            hypothesis = model(X.float())

            # Computation of the cost J
            cost = criterion(hypothesis, Y)  

            # Backward propagation
            cost.backward()  # <= compute the gradients

            # Update parameters (weights and biais)
            optimizer.step()

            # hardcoded batch size :( compute the train loss 
            train_loss += cost.data / 64
            acc = compute_accuracy(Y, hypothesis)
            if acc is not None:
                train_iou.append(acc)

        for batch_idx, (data, target) in enumerate(val_data):
            with torch.no_grad():
                # Send to device
                X = data.to(device)
                Y = target.to(device).float()

                # Forward pass on validation data
                hypothesis = model(X.float())

                # Compute val IOU
                acc = compute_accuracy(Y, hypothesis)
                if acc is not None:
                    val_iou.append(acc)
                
                # Computation of the loss J
                loss = criterion(hypothesis, Y)
                val_loss += loss.data / 64



            if batch_idx % 20 == 0:
                print("Epoch= {},\t batch = {},\t train loss = {:2.4f},\t train accuracy = {},\t val loss {},\t val accuracy".format(epoch + 1, batch_idx, train_loss, np.mean(train_iou.data.numpy()), val_loss, np.mean(val_iou.data.numpy())))


        mlflow.log_metric('Train IOU', np.mean(train_iou), epoch)
        mlflow.log_metric('Validation IOU',  np.mean(val_iou), epoch)
        mlflow.log_metric('Validation Loss',  train_loss, epoch)
        mlflow.log_metric('Training Loss',   val_loss, epoch)


device = torch.device("cuda:0")

# initialize model
run_name = 'Run 1'

with mlflow.start_run(run_name = run_name) as run:

    model = UNet(3, 1).to(device)

    # Params : batch_size, epochs, lr, epochs, currently batch size is unused 
    args = Params(32, 2, 0.001)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)


    train_loader = train_image_loader(img_files[201:], mask_files[201:])
    val_loader = train_image_loader(img_files[:200], mask_files[:200])

    train(model, train_loader, val_loader, args.lr, args.epochs, optimizer)

    # log the model once it is done training, then log the parameters
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    torch.save({
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'epoch':args.epochs
        }, 'run_stats.pyt')
    mlflow.log_artifact('run_stats.pyt')    