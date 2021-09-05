import argparse
import glob
import json
import multiprocessing
import sys
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

def progressLearning(value, endvalue, loss , acc , bar_length=50):
      
    percent = float(value + 1) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}/{2} \t Loss : {3:.3f} , Acc : {4:.3f}".format(arrow + spaces, value+1 , endvalue , loss , acc))
    sys.stdout.flush()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer) :

    lr = optimizer.param_groups[0]['lr']

    return lr

def train(data_dir, model_dir, args):
   
    seed_everything(args.seed)

    # -- settings / Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset / get module from dataset.py
    # class = args.dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation / get module from dataset.py
    # class = args.augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    # connect augmentation to dataset
    dataset.set_transform(transform)

    # split dataset
    train_set, val_set = dataset.split_dataset()

    # -- data loader 
    # train data loader
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- data loader
    # validation data loader
    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model / get module from model.py
    # class = args.model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    
    # -- optimizer
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )

    # -- scheduler
    scheduler_module = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.8)

    # -- logging
    logger = SummaryWriter(log_dir=args.log_dir)

    # training
    log_count = 0
    best_val_loss = np.inf

    for epoch in range(args.epochs):

        lr = get_lr(optimizer)
        print('Epoch : %d \t Learning Rate : %e'%(epoch,lr))

        model.train()
        for idx, train_batch in enumerate(train_loader):

            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)

            loss = criterion(outs, labels)
            acc = (torch.argmax(outs , dim=-1) == labels).float()
            acc = acc.mean()
            
            loss.backward()
            optimizer.step()

            progressLearning(idx , len(train_loader) , loss.item() , acc.item())

            if (idx + 1) % 10 == 0 :
                logger.add_scalar('Train/loss' , loss.item() , log_count)
                logger.add_scalar('Train/acc' , acc.item() , log_count)
                log_count += 1

        val_loss = 0.0
        val_acc = 0.0

        # validation
        with torch.no_grad():
            model.eval()
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)

                loss = criterion(outs, labels)
                acc = (torch.argmax(outs , dim=-1) == labels).float()
                acc = acc.mean()

                val_loss += loss
                val_acc += acc

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            if val_loss < best_val_loss:
       
                # update best val loss (min loss)
                best_val_loss = val_loss
            
                # save best model
                save_dir = args.model_dir    
                torch.save({'epoch' : (epoch) ,  
                            'model_state_dict' : model.state_dict() , 
                            'loss' : val_loss.item() , 
                            'acc' : val_acc.item()} , 
                            save_dir + '/checkpoint_model_bset.pt') 

            print('\nVal Loss : %.4f \t Val Acc : %.4f\n' %(val_loss , val_acc))

            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/acc", val_acc, epoch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')

    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')

    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/train/images')
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--log_dir' , type=str , default='./log')

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
