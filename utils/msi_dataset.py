from torch.utils.data import Dataset
import numpy as np
import random
import cv2
from PIL import Image
import torch
from torchvision import transforms


class HstackImg(Dataset):
    def read_txt(self, txt_file):
        data = []
        name = []
        with open(txt_file, encoding="utf-8") as fid:
            for line in fid:
                line = line.strip().split("\t")
                data.append(line[0])
                name.append(line[0].split('/')[-1])
        return data, name

    def __init__(self, txt_file, transform, color_transform=None):

        self.data, self.name = self.read_txt(txt_file)
        self.transform = transform
        self.color_transform = color_transform

    def __getitem__(self, index):
        res = {}
        img = cv2.imread(self.data[index])[:,:,::-1]
        rgb = img[:, :640, :]
        artery = img[:, 640:640 * 2, 0][:, :, np.newaxis] > 0
        vein = img[:, 640 * 2:640 * 3, 0][:, :, np.newaxis] > 0
        _570 = img[:, 640 * 3:640 * 4, 0][:, :, np.newaxis]
        _610 = img[:, 640 * 4:640 * 5, 0][:, :, np.newaxis]
        roi = img[:, 640 * 5:640 * 6, :]
        hyper = np.concatenate((_570, _610, np.zeros_like(_570)), axis=2)
        background = (1-artery*vein)==1
        vessel = np.concatenate((artery*255,vein*255,background*255), axis=2).astype(np.uint8)

        rgb = Image.fromarray(rgb)
        hyper = Image.fromarray(hyper)
        vessel = Image.fromarray(vessel)
        roi = Image.fromarray(roi)

        global_seed_state = torch.random.get_rng_state()
        with torch.random.fork_rng(devices=[]):
            seed = random.randint(-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff)
            if self.color_transform:
                torch.manual_seed(seed)
                color_jittered = self.color_transform(rgb)
                torch.manual_seed(seed)
                color_jittered = self.transform(color_jittered)
                res['color_jittered'] = color_jittered
            torch.manual_seed(seed)
            rgb = self.transform(rgb)
            torch.manual_seed(seed)
            hyper = self.transform(hyper)
            torch.manual_seed(seed)
            vessel = self.transform(vessel)
            torch.manual_seed(seed)
            roi = self.transform(roi)

        torch.random.set_rng_state(global_seed_state)

        vessel[vessel>0.5] = 1
        vessel[vessel<=0.5] = 0

        res['input'] = rgb
        res['output'] = hyper
        res ['vessel'] = vessel
        res['roi'] = roi
        return res

    def __len__(self):
        return len(self.data)

