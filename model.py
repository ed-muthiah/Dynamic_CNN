import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import *

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
            ### complete the forward path --------------------
            #
            #
        # out1=self.resnet18(imgs)
        # out2=self.embeds(labels)
        # out2=self.Weights_Generator_MLP(out2)
        # print(self.DC.weight.shape)
        # self.DC.weight=nn.Parameter(out2.expand(-1,64,-1,-1))
        # print(self.DC.weight.shape)
        # out1=self.DC(out1)
        # out1=out1.view(-1,64)
        # cls_scores=self.Class_Score_Predictor_MLP(out1)

        # ### ----------------------------------------------
        # return cls_scores # Dim: [batch_size, 10]
        
        cls_scores = torch.tensor((), dtype=torch.double, device=self.device)
        cls_scores = cls_scores.new_zeros((64, 10)) #64 is batch size
        for i in range(10):
            v = self.resnet18(imgs)
            w = self.Weights_Generator_MLP(v.flatten())
            w = F.normalize(w, p=2, dim=0)
            w = w.view_as(self.dc.weight.data)
            self.dc.weight.data = w
            v_hat = self.dc(v)
            v_hat = v_hat / np.sqrt(v_hat.size()[0])
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
