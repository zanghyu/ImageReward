'''
@File       :   CLIPScore.py
@Time       :   2023/02/12 13:14:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   CLIPScore.
* Based on CLIP code base
* https://github.com/openai/CLIP
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip
from CLIPScore import CLIPScore
import os

class HPSScore(CLIPScore):
    def __init__(self, download_root, device='cpu'):
        super(self).__init__(download_root, device)
        params = torch.load(os.path.join(download_root,'hpc.pt'))['state_dict']
        self.model.load_state_dict(params)

