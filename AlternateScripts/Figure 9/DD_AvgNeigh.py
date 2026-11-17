#Copyright 2022, Jason Lequyer and Laurence Pelletier, All rights reserved.
#Sinai Health SystemLunenfeld-Tanenbaum Research Institute
#600 University Avenue, Room 1070
#Toronto, ON, M5G 1X5, Canada
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tifffile import imread, imwrite
import sys
import numpy as np
from pathlib import Path
import time


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        #####Yize's fixes
        self.multi_channel = True
        self.return_mask = True
        
        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output






if __name__ == "__main__":
    functions = [lambda x: x, lambda x: np.rot90(x), lambda x: np.rot90(np.rot90(x)), lambda x: np.rot90(np.rot90(np.rot90(x))), lambda x: np.flipud(x), lambda x: np.fliplr(x), lambda x: np.rot90(np.fliplr(x)), lambda x: np.fliplr(np.rot90(x))]
    ifunctions = [lambda x: x, lambda x: np.rot90(x,-1), lambda x: np.rot90(np.rot90(x,-1),-1), lambda x: np.rot90(np.rot90(np.rot90(x,-1),-1),-1), lambda x: np.flipud(x), lambda x: np.fliplr(x), lambda x: np.rot90(np.fliplr(x)), lambda x: np.fliplr(np.rot90(x))]
    folder = sys.argv[1]
    outfolder = folder+'_DDAN'
    Path(outfolder).mkdir(exist_ok=True)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    


    class TwoCon(nn.Module):
    
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = PartialConv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
            self.conv2 = PartialConv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
            self.nonlinear1 = nn.LeakyReLU(0.1)
            self.nonlinear2 = nn.LeakyReLU(0.1)
    
        def forward(self, x, maskeroo):
            x, maskeroo = self.conv1(x, maskeroo)
            x = self.nonlinear1(x)
            x, maskeroo = self.conv2(x, maskeroo)
            x = self.nonlinear2(x)
            return x, maskeroo
    
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = TwoCon(1, 48)
            self.conv2 = TwoCon(48, 48)
            self.conv3 = TwoCon(48, 48)
            self.conv4 = TwoCon(48, 48)  
            self.conv5 = TwoCon(48, 48) 
            self.conv7 = TwoCon(48, 48) 
            self.conv6 = nn.Conv2d(48,1,1)
            
    
        def forward(self, x, maskero):
            x, maskero = self.conv1(x,maskero)
            x, maskero = self.conv2(x,maskero)
            x, maskero = self.conv3(x, maskero)
            x, maskero = self.conv4(x, maskero)
            x, maskero = self.conv5(x, maskero)
            x, maskero = self.conv7(x, maskero)
            x = torch.sigmoid(self.conv6(x))
            return x
        
    file_list = [f for f in os.listdir(folder)]
    
    for v in range(len(file_list)):
        
        
        file_name =  file_list[v]

        start_time = time.time()
        print(file_name)
        if file_name[0] == '.':
            continue
        
        img = imread(folder + '/' + file_name)
        
        typer = type(img[0,0])
        
        minner = np.amin(img)
        img = img - minner
        maxer = np.amax(img)
        img = img/maxer
        img = img.astype(np.float32)
        shape = img.shape
        
        
        
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        
        imgZ = imgZ[:,:]
        shape = imgZ.shape
        
        
        listimgV = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        
        
        imgin3 = imgZ.copy()
        imgin4 = imgZ.copy()
        for i in range(2,imgin3.shape[0]-2):
            for j in range(2,imgin3.shape[1]-2):
                otcount = [imgZ[i,j-1],imgZ[i,j+1],imgZ[i-1,j],imgZ[i+1,j]]
                if (i + j) % 2 == 0:
                    imgin3[i,j] = np.mean(otcount)
                else:
                    imgin4[i,j] = np.mean(otcount)
        imgin3 = torch.from_numpy(imgin3)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin = imgin3.to(device)
        imgin4 = torch.from_numpy(imgin4)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin2 = imgin4.to(device)
        
        

        
        augment = []
        for i in range(len(functions)):
            khufu = functions[i](img.copy())
            khufu = torch.from_numpy(khufu.copy())
            khufu = torch.unsqueeze(khufu,0)
            khufu = torch.unsqueeze(khufu,0).to(device)
            augment.append(khufu)

  
    
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img,0)
        img = torch.unsqueeze(img,0)
        img = img.to(device)

        #listimg = listimgV1+listimgV2+listimgV3+listimgV4
        
        net = Net()
        net.to(device)
        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(net.parameters(), lr=0.0005)

        bothap = imgin
        imgup =  imgin2
        bothapcpu = (bothap[0,0,:,:].cpu().detach().numpy()*maxer+minner)
        imgupcpu =  (imgup[0,0,:,:].cpu().detach().numpy()*maxer+minner)
        imgupcpunow = torch.from_numpy(imgupcpu).view(1,1,imgin.shape[2],imgin.shape[3])
        bothapcpunow = torch.from_numpy(bothapcpu).view(1,1,imgin2.shape[2],imgin2.shape[3])
        
        imgupcpunow = imgupcpunow.view(-1)
        bothapcpunow = bothapcpunow.view(-1)
        
        running_loss1=0.0
        last10mask = []
        last10blind = []
        for sp in range(len(ifunctions)):
            last10blind.append(functions[sp](np.zeros_like(img[0,0,:,:].cpu().detach().numpy())))
            last10mask.append(functions[sp](np.zeros_like(img[0,0,:,:].cpu().detach().numpy())))
        keepall = [np.zeros_like(bothap[0,0,:,:].cpu().detach().numpy()),
                   np.zeros_like(bothap[0,0,:,:].cpu().detach().numpy())]

        keepallcounter = 0
        keepallcounter2 = 0
        counter = 0
        goodo = True
        lastval = None
        newval = np.inf
        
        countso = 500
        
        last14img = [0]*37
        last9pct = [0]*15
        last10window = list(range(-30,0))
        
        maskert0 = torch.ones_like(bothap).to(device)

        while goodo:
            counter += 1
            switcher = np.random.randint(len(functions))
            
            tranimg = augment[switcher]
            
            masker = torch.rand(tranimg.shape)
            masker = torch.where(masker>0.8)
            maskerto = torch.ones_like(tranimg).to(device)
            maskerto[masker] = 0
            
            leaker = torch.rand(tranimg.shape)
            leaker = torch.where(leaker>0.999)
            leakerto = torch.zeros_like(tranimg).to(device)
            leakerto[leaker] = 1
            
            inputs = tranimg
            inputs = tranimg*maskerto
            labello = tranimg
            
            optimizer.zero_grad()
            outputs = net(inputs,maskerto)
            
            loss1 = torch.mean(((1.0-maskerto)+leakerto)*criterion(outputs, labello))
            
            
            if counter > countso:
                
                with torch.no_grad():

                    tempoutputs = outputs*maxer+minner
                    lasts2s =  torch.zeros_like(outputs).to(device)
                    lasts2s[torch.where(maskerto==1)] = tempoutputs[torch.where(maskerto==1)]
                    lasts2s = lasts2s[0,0,:,:].cpu().detach().numpy()
                    lastmasker = maskerto[0,0,:,:].cpu().detach().numpy()
                
                
                last10blind[switcher] += lasts2s
                last10mask[switcher] += lastmasker
                
                

            loss = loss1
            running_loss1+=loss1.item()
            loss.backward()
            optimizer.step()
            
            
            running_loss1=0.0

            if counter > countso:
            
                with torch.no_grad():
                    
                    
                    coinflip = counter % 2
                    if coinflip == 0:
                        imgdown = bothap
                        
                        outputs = net(imgdown,maskert0)
                        outputs = outputs[0,0,:,:].cpu().detach().numpy()
                        outputs = outputs*maxer+minner
                        keepall[0] += outputs
                        keepallcounter+=1
                    elif coinflip == 1:
                        imgdown = imgup
                        
                        outputs = net(imgdown,maskert0)
                        outputs = outputs[0,0,:,:].cpu().detach().numpy()
                        outputs = outputs*maxer+minner
                        keepall[1] += outputs
                        keepallcounter2+=1
                        
            if counter % countso == (countso-1) and counter > countso:
                
                
                H0 = keepall[0]/keepallcounter
                V0 = keepall[1]/keepallcounter2
                H0 = H0
                V0 = V0

                H0 = torch.from_numpy(H0).view(-1)
                V0 = torch.from_numpy(V0).view(-1)


                with torch.no_grad():
                    newval1 = ((imgupcpunow - H0)**2).cpu().detach().numpy()
                    newval2 = ((bothapcpunow - V0)**2).cpu().detach().numpy()

                
                newval = newval1+newval2
                

                if lastval is None:
                    lastval = newval
                curpct = (np.sum(newval>lastval)/len(newval>lastval))
                    
                lastval = newval
                
                last10blindfin = last10blind.copy()
                last10maskfin = last10mask.copy()
                for iw in range(len(ifunctions)):
                    last10blindfin[iw] = ifunctions[iw](last10blind[iw])
                    last10maskfin[iw] = ifunctions[iw](last10mask[iw])
                last10blindfin = np.sum(np.stack(last10blindfin), axis=0)
                last10maskfin = np.sum(np.stack(last10maskfin), axis=0)
                H2 = last10blindfin/last10maskfin
                #print(counter+1)
                
                last9pct.append(curpct)
                last9pct.pop(0)
                last14img.append(H2)
                last14img.pop(0)
                last10window.append(np.mean(last9pct))
                last10window.pop(0)
                
                if np.argmax(last10window) == 0:
                    goodo = False
                    H2 = last14img[0]
                
                keepall[0] = 0*keepall[0]
                keepall[1] = 0*keepall[1]
                keepallcounter = 0
                keepallcounter2 = 0
                
                    

                
        imwrite(outfolder + '/' + file_name, H2.astype(typer))
        
        print("--- %s seconds ---" % (time.time() - start_time)) 


        
        torch.cuda.empty_cache()