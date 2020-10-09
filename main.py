import argparse
import torch
import torchvision
import torchvision.transforms as transforms

from model import BaseModel
from train import train, resume, evaluate
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='df_0')
    parser.add_argument('--with_dyn', type=int, default=1, help='with/without using dynamic filters')

    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataloaders
    transform = transforms.Compose([transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainvalset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=False, transform=transform)
    trainset, testset = torch.utils.data.random_split(trainvalset, [49000, 1000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2,drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=2,drop_last=True)
    dataloaders = (trainloader, testloader)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # network
    
    model = BaseModel(args).to(device)
    print(model)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    
    # resume the trained model
    if args.resume:
        model, optimizer = resume(args, model, optimizer)

    if args.test == 1: # test mode, resume the trained model and test
        testing_accuracy = evaluate(args, model, testloader)
        print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))
    else: # train mode, train the network from scratch
        train(args, model, optimizer, dataloaders)
        print('training finished')
        print('starting image save')
        
        kernels = model.dc[0].weight.detach().clone()
        print('kernels shape,',kernels.size())
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()
        kernels_grid = make_grid(kernels)
        print('kernels_grid shape,',kernels.size())
        npgrid = kernels_grid.cpu().numpy()
        print('np grid shape,',npgrid.shape)
        npgrid_sq = np.squeeze(npgrid)
        print('np grid squeezed shape,',npgrid_sq.shape)
        plt.imsave('my_kernels.jpg',npgrid)
        print('image saved')
