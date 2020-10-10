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
        # Get full ResNet18 model
        self.resnet18 = models.resnet18(pretrained=True) 
        
        # Initialize our custom DCN model as the first 5 layers of ResNet up to BasicBlock (1)
        self.dcn_model = nn.Sequential(*list(self.resnet18.children())[:5]) 
        
        # Initialize the Weight Generator network
        self.w_dyn = nn.Sequential(
            nn.Linear(4096,576),
            nn.Tanh()
            )
        
        # Initialize the Dynamic Convolutional (DC) layer
        self.dc = nn.Sequential(
            nn.Conv2d(64,1,3,stride=1,padding=1)
            )
        
        # Initialize the weight generator network
        self.w_cls = nn.Sequential(
            nn.Linear(64,10),
            )
        #
        #
        ### ---------------------------------------------------------

    def forward(self, imgs,with_dyn=True):
        if with_dyn:
            ''' with dynamic convolutional layer '''
            # Notes for debugging
            # Batch Size was 64
            # imgs size was torch.Size([64, 3, 32, 32])
            # v size was torch.Size([64, 64, 8, 8])
            # w size was torch.Size([64, 576])
            # self.dc[0].weight.data size was torch.Size([1, 64, 3, 3])
            
            # Passing input imgs through backbone to get feature map v
            v = self.dcn_model(imgs)
            
            # Flatten feature map v and feed it to weights generator network
            w = self.w_dyn(v.view(imgs.size(0),-1))
            
            # L2 Normalise w along rows 
            w_normalised = F.normalize(w, p=2, dim=0)
            
            # Initialize class scores tensor on GPU
            cls_scores = torch.tensor((), dtype=torch.double, device=self.device)
            
            # Intialize class scores matrix shape Batch Size x No. Classes
            cls_scores = cls_scores.new_zeros((imgs.size(0), 10)) 

            # Iterate through all images in batch
            for img in range(imgs.size(0)):
                # Get the normalised dynamic network weights for each image and reshape
                w_img = w_normalised[img,:].view_as(self.dc[0].weight.data)

                #with torch.no_grad():
                # Set the weights of DC layer 
                self.dc[0].weight.data = w_img

                # Unsqueeze to make 3D into 4D for image feature map
                v_i = torch.unsqueeze(v[img,:, :, :], 0)
                
                # Feed feature map v_i through the DC layer
                v_hat_i = self.dc(v_i)
                
                # Normalise the image representation v_hat_i
                v_hat_i = v_hat_i / torch.norm(v_hat_i)
                
                # Reshape
                v_hat_i = v_hat_i.view(v_hat_i.size(0), imgs.size(0))         
                
                # Pass v_hat_i to the classification layer to get all 10 class scores
                cls_scores[[img],:] = self.w_cls(v_hat_i).double()
                #
                #
                ### ----------------------------------------------

        else:
            ''' without dynamic convolutional layer '''
            # To compare training stats to the same network without dynamic filters, 
            # we stop using the w_dyn network. This means that the network is simple
            # and only has 3 layers, DCN_model (from ResNet18) > DC layer > Classfication Layer
            
            # Pass imgs through our custom DCN Model
            output1=self.dcn_model(imgs)
            
            # Pass layer 1 output through DC layer
            output2=self.dc(output1)
            
            # Reshape and pass output of layer 2 through classification layer to get class scores
            cls_scores=self.w_cls(output2.view(-1,64))
            #
            ### ---------------------------------------------
        return cls_scores
