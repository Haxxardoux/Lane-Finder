from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# Size to resize incoming images to, both images and masks during training and validation
__C.new_size = (150, 200)

# batch size mainly for training, but also impacts dataloader for things like validation
__C.batch_size = 16

__C.lr = 0.001

__C.epochs = 2