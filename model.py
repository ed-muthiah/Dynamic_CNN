import torch
import torch.nn as nn
import numpy as np
import torchvision
import numpy as np
import os
import time
import sys
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, args):
        print(args)
        print(args.batch_size)
        print('hello')
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.class_num = 10

        ### initialize the BaseModel --------------------------------
        #
        #
        self.resnet18 = models.resnet18(pretrained=True)
        # self.net = self.net.cuda() if self.device else self.net
        self.dcn_model = nn.Sequential(*list(self.resnet18.children())[:5])
        self.w_dyn = nn.Sequential(
            nn.Linear(4096,576),
            nn.Tanh()
            )
        self.dc = nn.Sequential(
            nn.Conv2d(64,1,3,stride=2,padding=1)
            )
        self.w_cls = nn.Sequential(
            nn.Linear(64,10),
            )
        #
        #
        ### ---------------------------------------------------------

    def forward(self, imgs,with_dyn=True):
        if with_dyn:
            ''' with dynamic convolutional layer '''        
        cls_scores = torch.tensor((), dtype=torch.double, device=self.device)
        cls_scores = cls_scores.new_zeros((64, 10)) #64 is batch size
        v = self.dcn_model(imgs)
        print('imgs size is', imgs.size())
        print('v size is', v.size())
        w = self.w_dyn(v.view(imgs.size(0),-1))
        for i in range(v.size(0)):
            print('starting forward')
            w = F.normalize(w, p=2, dim=0)
            print('w size is', w.size())
            sys.exit()
            w = w.view_as(self.dc[0].weight.data)
            self.dc[0].weight.data = w
            v_hat = self.dc(v)
            v_hat = v_hat / torch.norm(v_hat)
            re_v_hat = v_hat.view(v_hat.size(0), 64)
            cls_scores[:, [i]] = self.w_cls(re_v_hat).double()
            #
            #
            ### ----------------------------------------------
            
        else:
            ''' without dynamic convolutional layer '''
            out1=self.resnet18(imgs)
            out1=self.dc(out1)
            out1=out1.view(-1,64)
            cls_scores=self.w_cls(out1)
            
            #
            ### ---------------------------------------------
        return cls_scores
