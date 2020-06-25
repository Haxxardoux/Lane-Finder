import torch
from torch.utils.data import DataLoader
import mlflow
from time import time
import hiddenlayer as HL

# i made all of these !
from video_loader import vidSet
from utils import Params, count_parameters
from loss import SupConLoss

# mlflow command
# mlflow ui --backend-store-uri Logs &


paths = []
paths.append('C:\\Users\\turbo\\Python projects\\Lane finder\\Lane finder\\videos\\Input_videos\\shadow_challenge.mp4')
# paths.append('C:\\Users\\turbo\\Python projects\\Lane finder\\Lane finder\\videos\\Input_videos\\obstacle_challenge.mp4')

# takes a list of file paths to .mp4s and returns a dataloader ov the frames
vidset_train = vidSet(paths)

# we want a class for our parameters because it is wayyyy easier to log them this way 
args = Params(15, 2, 0.0001)

vidloader_train = DataLoader(vidset_train, batch_size=args.batch_size, shuffle=False)

# set the file path where the logs will be stored. this should be a global reference since many different scripts will reference it from different directories
mlflow.tracking.set_tracking_uri('file:\\Users\\turbo\\Python projects\\Lane finder\\Logs')

# a new experiment will be created if one by that name does not already exists
mlflow.set_experiment('Triplet loss unsupervised')

device = 'cuda:0'

def train(model, train_data, loss, optimizer):

    for epoch in range(args.epochs):
        t0 = time()
        train_loss = 0

        for batch_idx, img_tensor in enumerate(train_data):

            # select FOUR images total, two pos two neg. the forward pass size is 1 gb for the overparameterized model so it is important
            # to pick a combination that fits into memory. 
            
            # Select 2 positives, or the first 2 frames
            positives_tensor = img_tensor[:2].to(device)
            
            # this I THINK selects the last two images in a sequence, loss does not count observations of the same class, so it is ok that they are similar
            # however it may be slower to learn.
            negatives_tensor = img_tensor[(len(img_tensor)-2):].to(device)

            # print('shape of two positives array: ', positives_tensor.shape)
            # print('shape of two negatives array: ', negatives_tensor.shape)

            # set parameter gradients to zero 
            optimizer.zero_grad()

            # Forward pass: compute the output
            positive_latent_tensor = model(positives_tensor)
            negative_latent_tensor = model(negatives_tensor)

            # Computation of the cost J
            cost = loss(positive_latent_tensor, negative_latent_tensor)  

            # Backward propagation
            cost.backward()  # <= compute the gradients

            # Update parameters (weights and biais)
            optimizer.step()

            # hardcoded batch size :( compute the train loss 

            train_loss += cost.item()
        t1 = time()

        
        ret = {'Train Loss':train_loss, 'Epoch time':t1-t0}
        yield ret.items()


# load the model
from models import Backbone as Model

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

run_name = 'One video'
with mlflow.start_run(run_name = run_name) as run:
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    mlflow.log_param('Parameters', count_parameters(model))

    for epoch, items in enumerate(train(model, vidloader_train, SupConLoss(), optimizer)):
        for key, value in items:
            print(key, value)
            mlflow.log_metric(key, value, epoch)

    torch.save({
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        }, 'run_stats.pyt')
    mlflow.log_artifact('run_stats.pyt')

    torch.cuda.empty_cache()

    # save an architecture diagram
    HL.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
    HL.build_graph(model, torch.zeros([args.batch_size, 3, 288, 512]).to(device)).save('architecture', format='png')
    mlflow.log_artifact('architecture.png')