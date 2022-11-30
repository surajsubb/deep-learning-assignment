import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import Dataloader
import Model
import loss
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import time
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader
import shutil
import glob
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Enhancenet(nn.Module):
    """Solver for EnhanceNet"""
    def __init__(self, args):
        super(Enhancenet, self).__init__()
        self.lowlight_images_path=args.lowlight_images_path
        self.lr=args.lr
        self.weight_decay=args.weight_decay
        self.grad_clip_norm=args.grad_clip_norm
        self.num_epochs=args.num_epochs
        self.train_batch_size=args.train_batch_size
        self.val_batch_size=args.val_batch_size
        self.num_workers=args.num_workers
        self.display_iter=args.display_iter
        self.snapshot_iter=args.snapshot_iter


        self.load_pretrain=args.load_pretrain

        self.test=args.test
        # Other configurations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configuration for Directories
        self.snapshots_folder=args.snapshots_folder
        self.pretrain_dir=args.pretrain_dir
        self.Logger_folder=args.Logger_folder
        self.eval_folder=args.enhanced_dir
        self.writer=SummaryWriter(self.Logger_folder)

        # Build model
        self.build()


    def build(self):
        self.train_dataset = Dataloader.lowlight_loader(self.lowlight_images_path)		
        self.test_dataset=Dataloader.lowlight_test_loader(self.lowlight_images_path)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.test_loader=torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.model=Model.EnhanceNet()
        if self.load_pretrain == True:
	        self.model.load_state_dict(torch.load(self.pretrain_dir))
        else:
            self.model.apply(weights_init)
        self.model.to(self.device)
        
        self.L_color = loss.L_color()
        self.L_spa = loss.L_spa()
        self.L_exp = loss.L_exp(16,0.6)
        self.L_TV = loss.L_TV()
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.test:
            self.model.eval()
        else:
            self.model.train()      
    
    def train(self):
        
        for epoch in range(self.num_epochs):
            losses=[]     
            for iteration, img_lowlight in enumerate(self.train_loader):     
                img_lowlight = img_lowlight.to(self.device)      
                enhanced_image_1,enhanced_image,A  = self.model(img_lowlight)        
                Loss_TV = 200*self.L_TV(A)      
                loss_spa = torch.mean(self.L_spa(enhanced_image, img_lowlight))      
                loss_col = 5*torch.mean(self.L_color(enhanced_image))
                loss_exp = 10*torch.mean(self.L_exp(enhanced_image))        
                # best_loss
                loss =  Loss_TV + loss_spa + loss_col + loss_exp
            	        
                losses.append(loss.item())
            	
                self.optimizer.zero_grad()
            	
                loss.backward()
            	
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.grad_clip_norm)
                self.optimizer.step()
                if ((iteration+1) % self.display_iter) == 0:
                	print("Loss at iteration", iteration+1, ":", loss.item())
                if ((iteration+1) % self.snapshot_iter) == 0:
                
                	torch.save(self.model.state_dict(), self.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		
            total_loss=sum(losses)/len(losses)
            self.writer.add_scalar("Loss/train", total_loss, epoch)

    def Test(self):
        with torch.no_grad():
            for iteration,(image_path,img_lowlight) in enumerate(self.test_loader):
                img_lowlight = img_lowlight.to(self.device)
                _,enhanced_image,_  = self.model(img_lowlight)
                image=image_path[0].split("/")[-1]
                print(image)
                image=image.split('.')[0]

                result_path = os.path.join(self.eval_folder,image+".png")
                torchvision.utils.save_image(enhanced_image, result_path)



