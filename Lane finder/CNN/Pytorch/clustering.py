import torch
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# clustering 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# my stuff
from dataloader import image_loader
from model_components import UNet

# load the trained semantic segmentation network
device = torch.device('cuda:0')
net = UNet(3, 1, bilinear=True).to(device)
net.load_state_dict(torch.load('C://Users//turbo//Python projects//Lane-finder//Lane-Finder//CNN//checkpoints//CONST_NUM_FILCP_epoch7.pth', map_location=device))
net.eval()

# load the dataset, the training set is the first 1000 images, test set is remaining 1000
img_files = []
for file in glob.glob(os.path.abspath('C:/Users/turbo/Python projects/Lane-finder/Lane-Finder/data/imgs/*.png')):
    img_files.append(file)

mask_files = []
for file in glob.glob(os.path.abspath('C:/Users/turbo/Python projects/Lane-finder/Lane-Finder/data/masks/*.png')):
    mask_files.append(file)

data_loader = image_loader(img_files, mask_files)

for batch_idx, (data, target) in enumerate(data_loader):
    sample = target.reshape(target.shape[2:])

    # temp
    img = 255*sample.reshape(1, *sample.shape).numpy().transpose(1, 2, 0)
    img = np.dstack( (img, img, img) )
    plt.imshow(img)
    plt.show()

    db = DBSCAN(eps=.35, min_samples=100)
    features = StandardScaler().fit_transform(sample)
    db.fit(features)

    db_labels = db.labels_
    unique_labels = np.unique(db_labels)

    num_clusters = len(unique_labels)
    cluster_centers = db.components_

    ret = {
        'origin_features': features,
        'cluster_nums': num_clusters,
        'db_labels': db_labels,
        'unique_labels': unique_labels,
        'cluster_center': cluster_centers
    }
    print(ret)
    break
