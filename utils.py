import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ImageSplitter:
    def __init__(self, seg_size=48, scale=2, pad_size=3):
        self.seg_size = seg_size
        self.scale = scale
        self.pad_size = pad_size
        self.height = 0
        self.width = 0

    def split(self, pil_img):
        img_tensor = TF.to_tensor(pil_img).unsqueeze(0)
        _, _, h, w = img_tensor.size()
        self.height = h
        self.width = w

        pad_h = (h // self.seg_size + 1) * self.seg_size - h
        pad_w = (w // self.seg_size + 1) * self.seg_size - w

         # make sure the image is divisible into regular patches
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), 'reflect')

        # add padding around the image to simplify computations
        img_tensor = F.pad(img_tensor, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), 'reflect')

        _, _, h, w = img_tensor.size()
        self.height_padded = h
        self.width_padded = w

        patches = []
        for i in range(self.pad_size, h-self.pad_size, self.seg_size):
            for j in range(self.pad_size, w-self.pad_size, self.seg_size):
                patch = img_tensor[:, :,
                    (i-self.pad_size):min(i+self.pad_size+self.seg_size, h),
                    (j-self.pad_size):min(j+self.pad_size+self.seg_size, w)]
                patches.append(patch)

        return patches

    def merge(self, patches):
        pad_size = self.scale * self.pad_size
        seg_size = self.scale * self.seg_size
        height = self.scale * self.height
        width = self.scale * self.width
        height_padded = self.scale * self.height_padded
        width_padded = self.scale * self.width_padded

        out = torch.zeros((1, 3, height_padded, width_padded))
        patch_tensors = copy.copy(patches)

        for i in range(pad_size, height_padded-pad_size, seg_size):
            for j in range(pad_size, width_padded-pad_size, seg_size):
                patch = patch_tensors.pop(0)
                patch = patch[:, :, pad_size:-pad_size, pad_size:-pad_size]

                _, _, h, w = patch.size()
                out[:, :, i:i+h, j:j+w] = patch

        out = out[:, :, pad_size:height+pad_size, pad_size:width+pad_size]

        return TF.to_pil_image(out.clamp(0,1).squeeze(0))
