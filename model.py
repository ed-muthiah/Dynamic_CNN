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
        print('Arguements were:',args)
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.class_num = 10

        ### initialize the BaseModel --------------------------------
        #
        #
        self.resnet18 = models.resnet18(pretrained=True)
        # self.resnet18 = self.net.cuda() if self.device else self.net
        self.dcn_model = nn.Sequential(*list(self.resnet18.children())[:5])
        self.w_dyn = nn.Sequential(
            nn.Linear(4096,576),
            nn.Tanh()
            )
        self.dc = nn.Sequential(
            nn.Conv2d(64,1,3,stride=1,padding=1)
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
            #print('with dynamic convolutional layer')
            cls_scores = torch.tensor((), dtype=torch.double, device=self.device)
            cls_scores = cls_scores.new_zeros((64, 10)) #64 is batch size
            v = self.dcn_model(imgs)
            #print('imgs size is', imgs.size())
            #print('v size is', v.size())
    #imgs size is torch.Size([64, 3, 32, 32])
    #v size is torch.Size([64, 64, 8, 8])
    #w size is torch.Size([64, 576])
    #self.dc[0].weight.data size is torch.Size([1, 64, 3, 3])

            w = self.w_dyn(v.view(imgs.size(0),-1))
            w_normalised = F.normalize(w, p=2, dim=0)
            #print('w size is', w.size())
            for img in range(v.size(0)):
                #print('starting forward')
                #print('self.dc[0].weight.data size is', self.dc[0].weight.data.size())
                w_img = w_normalised[img,:].view_as(self.dc[0].weight.data)
                with torch.no_grad():
                    self.dc[0].weight.data = w_img

                v_i = torch.unsqueeze(v[img,:, :, :], 0)
                #print('v_i', v_i.size())

                v_hat_i = self.dc(v_i)
                v_hat_i = v_hat_i / torch.norm(v_hat_i)
                v_hat_i = v_hat_i.view(v_hat_i.size(0), 64)
                cls_scores[[img],:] = self.w_cls(v_hat_i).double()
                #print('done wooohooo')
                #sys.exit()

                #
                #
                ### ----------------------------------------------

        else:
            ''' without dynamic convolutional layer '''
            #print('withOUT dynamic convolutional layer')
            out1=self.dcn_model(imgs)
            out1=self.dc(out1)
            out1=out1.view(-1,64)
            cls_scores=self.w_cls(out1)
            
            #
            ### ---------------------------------------------
        return cls_scores
