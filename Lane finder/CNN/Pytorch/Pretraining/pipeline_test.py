import torch
from torch.utils.data import DataLoader
import mlflow
from time import time
import hiddenlayer as HL
import os
import cv2

# i made all of these !
from models import Backbone as Model
from video_loader import vidSet
from utils import Params, count_parameters
from loss import ConstrastiveLoss

# mlflow command
# mlflow ui --backend-store-uri Logs &


videos_path = 'C:\\Users\\turbo\\Python projects\\Lane finder\\data\\videos\\test'

path_list = []
for (dirpath, _, filenames) in os.walk(videos_path):
    for filename in filenames:
        path_list.append(os.path.abspath(os.path.join(videos_path, filename)))

# takes a list of file paths to .mp4s and returns a dataloader ov the frames
vidset_train = vidSet(path_list[:2])

# we want a class for our parameters because it is wayyyy easier to log them this way 
args = Params(1, 1, 0.00005)

vidloader_train = DataLoader(vidset_train, batch_size=args.batch_size, shuffle=False)

# set the file path where the logs will be stored. this should be a global reference since many different scripts will reference it from different directories
mlflow.tracking.set_tracking_uri('file:\\Users\\turbo\\Python projects\\Lane finder\\Logs')

# a new experiment will be created if one by that name does not already exists
mlflow.set_experiment('Constrastive loss unsupervised')

device = 'cuda:0'

# requires a custom training loop. if you are reading this and see that i use custom training loops in all my code, it is because as of right now,
# i really hope i can stop doing that and make my life easier. if you see custom training loops elsewhere, take pleasure in knowing that i failed, and i probably forgot 
def train(model, train_data, loss, optimizer):

    for epoch in range(args.epochs):
        t0 = time()
        train_loss = 0

        # this will be a tensor of 15 latent vectors corresponding to 15 successive video frames, which will sit in GPU memory and be used to compute loss
        # the issue is the latent vectors should change with training, and the only way to account for this is to re-compute them, but we ignore this!
        # we need to accumulate some latent vectors in memory. i think that if you start training with a decent model, or use a low lr, this is ok
        # how dare you point out an obvious flaw in my method!

        # torch.cat is too memory intensive - it creates copies, so we have to use a list instead
        # but this means we now have to transfer our latent tensor to GPU/host all the god damn time
        # :(

        # see above ^ not in use currently
        # latent_tensor = torch.zeros(15, 128).to(device).requires_grad_()
        latent_tensor = []

        for batch_idx, img in enumerate(train_data):

            # set parameter gradients to zero 
            optimizer.zero_grad()
            
            # take new observation to the gpu
            img_gpu = img.to(device)

            # concatenate the latent representation of the image with the latent tensor residing in GPU memory
            print('batch number: ', batch_idx)
            if batch_idx < 2:
                # not in use see above paragraph
                # latent_tensor = torch.cat( (latent_tensor[1:], model(img_gpu)), dim=0)

                # list instead, no drop op yet because we wait until we have 15 observations
                with torch.no_grad():
                    latent_vector = model(img_gpu)
                    latent_tensor.append(latent_vector)
                    print('no grad tensor shpae', latent_tensor[0].shape)
                    continue 
            else:
                # not in use, see above paragraph
                # latent_tensor = torch.cat( (latent_tensor[1:], model(img_gpu)), dim=0)

                # list instead
                latent_vector = model(img_gpu)
                latent_tensor.append(latent_vector)
                del latent_tensor[0]
                print('requires grad tensor shape', latent_tensor[0].shape)


            # Computation of the cost 
            # first argument is positive tensor and second argument is negative tensor, although it may not matter?
            # most recent ( [-1] ) observation in latent tensor is most recent, aka the "anchor" in the case of triplet loss

            # the 2 here (in both cases) corresponds to the number of positives/negatives. [-1] is the anchor, [-2:] is one anchor and one positive 
            cost = loss(latent_tensor[-2:], latent_tensor[:2])

            # Backward propagation
            cost.backward(retain_graph=True)  # <= compute the gradients

            # Update parameters (weights and biais)
            optimizer.step()

            # hardcoded batch size :( compute the train loss 

            train_loss += cost.item()
        t1 = time()

        
        ret = {'Train Loss':train_loss, 'Epoch time':t1-t0}
        yield ret.items()



model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
criterion = ConstrastiveLoss()
run_name = 'delete'
with mlflow.start_run(run_name = run_name) as run:
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    mlflow.log_param('Parameters', count_parameters(model))

    for epoch, items in enumerate(train(model, vidloader_train, criterion, optimizer)):
        for key, value in items:
            print('Epoch: ', epoch)
            print(key, value)
            mlflow.log_metric(key, value, epoch)

        torch.save({
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            }, str(epoch)+'run_stats.pyt')
        mlflow.log_artifact(str(epoch)+'run_stats.pyt')


    # torch.cuda.empty_cache()
    # # save an architecture diagram
    # HL.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
    # HL.build_graph(model, torch.zeros([args.batch_size, 3, 288, 512]).to(device)).save('architecture', format='png')
    # mlflow.log_artifact('architecture.png')