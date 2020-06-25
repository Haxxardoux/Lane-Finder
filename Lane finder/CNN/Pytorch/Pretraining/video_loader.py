import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

class vidSet(Dataset):
    def __init__(self, video_paths):
        self.video_paths = video_paths

        self.caps = [cv2.VideoCapture(video_path) for video_path in self.video_paths]
        self.images = [[capid, framenum] for capid, cap in enumerate(self.caps) for framenum in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-15))]    
        print(self.images)

    def __len__(self):
         return len(self.images)

    def __getitem__(self, idx):
        capid, framenum = self.images[idx]
        cap = self.caps[capid]
        cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
        res, frame = cap.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
        img_tensor = F.interpolate(img_tensor, scale_factor=(0.4, 0.4))

        return img_tensor.squeeze()

if __name__ == "__main__":
    import os 

    videos_path = 'C:\\Users\\turbo\\Python projects\\Lane finder\\data\\videos\\test'

    path_list = []
    for (dirpath, _, filenames) in os.walk(videos_path):
        for filename in filenames:
            path_list.append(os.path.abspath(os.path.join(videos_path, filename)))

    vidSet(path_list[:2])




