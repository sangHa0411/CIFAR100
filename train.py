import os
import sys
import random
import argparse
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import ResNet

def progressLearning(value, endvalue, loss, acc, bar_length=50):
    percent = float(value + 1) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r[{0}] {1}/{2} \t Loss : {3:.3f} , Acc : {4:.3f}".format(arrow + spaces, value+1, endvalue, loss, acc))
    sys.stdout.flush()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args) :
    # -- Seed
    seed_everything(args.seed)

    # -- Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- Data Augmentation
    img_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Resize(args.img_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))]
    )

    # -- Dataset
    train_dset = torchvision.datasets.CIFAR100(root='./Data', 
        train=True,
        download=True, 
        transform=img_transform
    )
    test_dset = torchvision.datasets.CIFAR100(root='./Data', 
        train=False,
        download=True, 
        transform=img_transform
    )

    # -- Dataloader
    train_loader = DataLoader(train_dset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=multiprocessing.cpu_count()//2
    )
    test_loader = DataLoader(test_dset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=multiprocessing.cpu_count()//2
    )

    # -- Model Specification
    assert args.model in [18,34]
    if args.model == 34 :
        layer_dim = [2, 3, 5, 2]
        ch_dim = [64, 128, 256, 512]
    else : # args.model == 18
        layer_dim = [1, 1, 1, 1]
        ch_dim = [64, 128, 256, 512]
    start_kernal = 7
    kernal_size = 3

    # -- Model
    model = ResNet(layer_list = layer_dim , 
        image_size=args.img_size , 
        ch_list = ch_dim , 
        in_kernal = start_kernal , 
        kernal_size = kernal_size , 
        class_size=100
    ).to(device)
    
    # -- Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9)

    # -- Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iteration, eta_min=args.min_lr)
    
    # -- Logging
    writer = SummaryWriter(args.log_dir)

    # -- Criterion 
    criterion = nn.CrossEntropyLoss().to(device)

    # -- Training
    min_loss = np.inf
    stop_count = 0
    log_count = 0
    # for each epoch
    for epoch in range(args.epochs) :
        idx = 0
        model.train()
        print('Epoch : %d/%d \t Learning Rate : %e' %(epoch, args.epochs, optimizer.param_groups[0]["lr"]))
        # training process
        for img_data, img_label in train_loader :
            img_data = img_data.float().to(device)
            img_label = img_label.long().to(device)

            optimizer.zero_grad()

            img_out = model(img_data)
        
            loss = criterion(img_out, img_label)
            acc = (torch.argmax(img_out, dim=-1) == img_label).float().mean()

            loss.backward()
            optimizer.step()
        
            progressLearning(idx, len(train_loader), loss.item(), acc.item())

            if (idx + 1) % 10 == 0 :
                writer.add_scalar('train/loss', loss.item(), log_count)
                writer.add_scalar('train/acc', acc.item(), log_count)
                log_count += 1
            idx += 1

        # validation process
        with torch.no_grad() :
            model.eval()
            loss_eval = 0.0
            acc_eval = 0.0
            for img_data, img_label in test_loader :
                img_data = img_data.float().to(device)
                img_label = img_label.long().to(device)

                img_out = model(img_data)
        
                loss_eval += criterion(img_out, img_label)
                acc_eval += (torch.argmax(img_out, dim=-1) == img_label).float().mean()
            loss_eval /= len(test_loader)
            acc_eval /= len(test_loader)

        writer.add_scalar('test/loss', loss_eval.item(), epoch)
        writer.add_scalar('test/acc', acc_eval.item(), epoch)
    
        if loss_eval < min_loss :
            min_loss = loss_eval
            torch.save({'epoch' : (epoch) ,  
                'model_state_dict' : model.state_dict() , 
                'loss' : loss_eval.item() , 
                'acc' : acc_eval.item()} , 
                os.path.join(args.model_dir, 'resnet_cifar100.pt'))        
            stop_count = 0 
        else :
            stop_count += 1
            if stop_count >= 5 :      
                print('\tTraining Early Stopped')
                break
            
        scheduler.step()
        print('\nTest Loss : %.3f \t Test Accuracy : %.3f\n' %(loss_eval, acc_eval))
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--img_size', type=int, default=224, help='input image size (default: 224)')
    parser.add_argument('--model', type=int, default=34, help='model layer size of resnet(18 or 34)')
    parser.add_argument('--iteration', type=int, default=5, help='max iteration of cosine annealing scheudler')
    parser.add_argument('--init_lr', type=float, default=2.5e-4, help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='minimum learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    # Container environment
    parser.add_argument('--model_dir', type=str, default='./Model')
    parser.add_argument('--log_dir' , type=str , default='./Log')

    args = parser.parse_args()
    train(args)
