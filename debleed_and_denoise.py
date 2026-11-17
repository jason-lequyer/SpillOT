#Copyright 2024, Jason Lequyer and Laurence Pelletier, All rights reserved.
#Sinai Health SystemLunenfeld-Tanenbaum Research Institute
#600 University Avenue, Room 1070
#Toronto, ON, M5G 1X5, Canada
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tifffile import imread, imwrite, TiffFile
import sys
import numpy as np
from pathlib import Path
import time
from scipy.sparse import coo_matrix
import scipy.sparse.csgraph as ssc
from collections import Counter
from scipy.optimize import linear_sum_assignment
import pandas as pd







if __name__ == "__main__":
    

    
    functions = [lambda x: x, lambda x: np.rot90(x), lambda x: np.rot90(np.rot90(x)), lambda x: np.rot90(np.rot90(np.rot90(x))), lambda x: np.flipud(x), lambda x: np.fliplr(x), lambda x: np.rot90(np.fliplr(x)), lambda x: np.fliplr(np.rot90(x))]
    ifunctions = [lambda x: x, lambda x: np.rot90(x,-1), lambda x: np.rot90(np.rot90(x,-1),-1), lambda x: np.rot90(np.rot90(np.rot90(x,-1),-1),-1), lambda x: np.flipud(x), lambda x: np.fliplr(x), lambda x: np.rot90(np.fliplr(x)), lambda x: np.fliplr(np.rot90(x))]
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FLOAT32_MAX = 3.402823466E+38
    FLOAT32_MAX_SQRT = np.sqrt(3.402823466E+38)
    FLOAT32_MAX_4RT = np.sqrt(np.sqrt(3.402823466E+38))
    FLOAT32_MAX_8RT = np.sqrt(np.sqrt(np.sqrt(3.402823466E+38)))
    FLOAT32_MAX_16RT = np.sqrt(FLOAT32_MAX_8RT)
    FLOAT32_MAX_32RT = np.sqrt(FLOAT32_MAX_16RT)
    
    FLOAT32_MIN = 1.175494351E-38
    patsize = 16
    
    varkern = torch.from_numpy(np.array([[0,1,0],[1,-4,1],[0,1,0]]))
    
        
    #ParticalConv2D is copied from https://github.com/JinYize/self2self_pytorch, which is based on https://github.com/NVIDIA/partialconv
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
            x = self.conv6(x)
            return x
        

    file_name =  sys.argv[1]
    filenoext = os.path.splitext(file_name)[0]

    print(file_name)
    
    inp = imread(file_name)
    

    if inp.shape[-1] == 3:
        inp = np.swapaxes(inp, -2, -1)
        inp = np.swapaxes(inp, -3, -2)
    ogshape = inp.shape
    
    inp = inp.reshape(-1,ogshape[-2],ogshape[-1])
    
    typer = type(inp[0,0,0])
    
    inpnorm = inp.copy().astype(np.float32)
    
    out = inpnorm.copy()

    inpnorm = torch.from_numpy(inpnorm)
    
    
    path_file_name = Path(file_name)

    oz = int(sys.argv[2]) - 1

        
    start_time = time.time()
    
    base2d = torch.clone(inpnorm[oz,:,:])

    #colum = (np.append(colum[0],oz))
    compare3d = torch.clone(inpnorm)
    

        

    
    ogshapo = base2d.shape
    
    
    
    
    
    with torch.no_grad():
    
        if len(base2d.shape) == 2:
            base2d =  torch.unsqueeze(base2d,0)
        base2d =  torch.unsqueeze(base2d,0)
        
        if len(compare3d.shape) == 2:
            compare3d =  torch.unsqueeze(compare3d,0)
        compare3d =  torch.unsqueeze(compare3d,0)
        
        curcomp = torch.clone(compare3d[:,oz:oz+1,:,:])
        
        try:
            matrix = pd.read_csv(file_name.split('.')[0]+'.csv',index_col=0).fillna(1)
            cur_rwo = matrix.iloc[oz]
            included = np.where(cur_rwo == 1)
            compare3d = compare3d[:,included[0],:,:]
        except:
            print('No matrix found, using all channels for bleed through removal.')

        
        dummy = torch.ones_like(base2d)
        
    
        
        
        unfold = nn.Unfold(kernel_size=(patsize+4, patsize+4))
        RP = nn.ReflectionPad2d((2, 2, 2, 2))
        base2d = RP(base2d)
        dummy = RP(dummy)
        
        
        rflef = torch.zeros_like(base2d)
        rflef[0,0,2:-2,1] = 2
        rflef[0,0,2:-2,2] = 1
        rfrit = torch.zeros_like(base2d)
        rfrit[0,0,2:-2,-2] = 2
        rfrit[0,0,2:-2,-3] = 1
        rftop = torch.zeros_like(base2d)
        rftop[0,0,1,2:-2] = 2
        rftop[0,0,2,2:-2] = 1
        rfbot = torch.zeros_like(base2d)
        rfbot[0,0,-2,2:-2] = 2
        rfbot[0,0,-3,2:-2] = 1
        
        
    
        
        evens = torch.zeros((patsize+4,patsize+4))
        odds = torch.zeros((patsize+4,patsize+4))
        
        for i in range(1,evens.shape[0]-1):
            for j in range(1,evens.shape[1]-1):
                if i == 1 or i == (evens.shape[0]-2) or j == 1 or j == (evens.shape[1]-2):
                    if (i + j) % 2 == 0:
                        evens[i,j] = 1
                    else:
                        odds[i,j] = 1
                else:
                    continue
        rf = torch.cat([rflef,rfrit,rftop,rfbot],axis=0)
        base2d = unfold(base2d)
        dummy = unfold(dummy)
        rf = unfold(rf)
        
        
    
        
        odds = torch.where(odds>0)
        evens = torch.where(evens>0)
    
    
        
        base2d = base2d.view(-1,patsize+4,patsize+4,base2d.shape[-1])
        dummy = dummy.view(-1,patsize+4,patsize+4,base2d.shape[-1])
        rf = rf.view(-1,patsize+4,patsize+4,base2d.shape[-1])
        dom2d = torch.clone(base2d)
    
        temp = torch.clone(rf[0,2:-2,1:-1,:])
        rf[0,:,:,:] = 0*rf[0,:,:,:]
        rf[0,2:-2,1:-1,:] = torch.clone(temp)
    
        temp = torch.clone(rf[1,2:-2,1:-1,:])
        rf[1,:,:,:] = 0*rf[1,:,:,:]
        rf[1,2:-2,1:-1,:] = torch.clone(temp)
    
        temp = torch.clone(rf[2,1:-1,2:-2,:])
        rf[2,:,:,:] = 0*rf[2,:,:,:]
        rf[2,1:-1,2:-2,:] = torch.clone(temp)
        
        temp = torch.clone(rf[3,1:-1,2:-2,:])
        rf[3,:,:,:] = 0*rf[3,:,:,:]
        rf[3,1:-1,2:-2,:] = torch.clone(temp)
        
        top = torch.zeros((patsize+4,patsize+4,patsize+4,patsize+4))
        bot = torch.zeros((patsize+4,patsize+4,patsize+4,patsize+4))
        lef = torch.zeros((patsize+4,patsize+4,patsize+4,patsize+4))
        rit = torch.zeros((patsize+4,patsize+4,patsize+4,patsize+4))
        toplef = torch.zeros((patsize+4,patsize+4,patsize+4,patsize+4))
        toprit = torch.zeros((patsize+4,patsize+4,patsize+4,patsize+4))
        botlef = torch.zeros((patsize+4,patsize+4,patsize+4,patsize+4))
        botrit = torch.zeros((patsize+4,patsize+4,patsize+4,patsize+4))
        cn = torch.zeros((patsize+4,patsize+4,patsize+4,patsize+4))
        
    
        bord = torch.zeros((patsize+4,patsize+4,patsize+4,patsize+4))
        
    
        
        
        for ka in range(1,top.shape[0]-1):
            for kb in range(1,top.shape[1]-1):
                top[ka,kb,ka-1,kb] = 1
                bot[ka,kb,ka+1,kb] = 1
                lef[ka,kb,ka,kb-1] = 1
                rit[ka,kb,ka,kb+1] = 1
                toplef[ka,kb,ka-1,kb-1] = 1
                toprit[ka,kb,ka-1,kb+1] = 1
                botlef[ka,kb,ka+1,kb-1] = 1
                botrit[ka,kb,ka+1,kb+1] = 1
                cn[ka,kb,ka,kb] = 1
                if ka == 1 or ka == (top.shape[0]-2) or kb == 1 or kb == (top.shape[1]-2):
                    if (ka + kb) % 2 == 0:
                        bord[ka,kb][odds] = 1
                    else:
                        bord[ka,kb][evens] = 1
                if ka == 1:
                    bord[ka,kb,ka+1,kb] = 1
                if kb == 1:
                    bord[ka,kb,ka,kb+1] = 1
                if ka == (top.shape[0]-2):
                    bord[ka,kb,ka-1,kb] = 1
                if kb == (top.shape[1]-2):
                    bord[ka,kb,ka,kb-1] = 1
                
        
        top = torch.where(top==1)
        bot = torch.where(bot==1)
        lef = torch.where(lef==1)
        rit = torch.where(rit==1)
        toplef = torch.where(toplef==1)
        toprit = torch.where(toprit==1)
        botlef = torch.where(botlef==1)
        botrit = torch.where(botrit==1)
        cn = torch.where(cn==1)
        bord = torch.where(bord==1)
        
        
        
    
        for hj in range(base2d.shape[-1]):
            curpat = torch.clone(base2d[0,:,:,hj])
            
            
            curpat = curpat.repeat(patsize+4,patsize+4,1,1)
            
        
            
            
            curcost = torch.zeros_like(curpat)
            
            curcost[top] = torch.abs((curpat[toprit] - curpat[rit])) + FLOAT32_MIN
            curcost[bot] = torch.abs((curpat[botlef] - curpat[lef])) + FLOAT32_MIN
            curcost[lef] = torch.abs((curpat[toplef] - curpat[top])) + FLOAT32_MIN
            curcost[rit] = torch.abs((curpat[botrit] - curpat[bot])) + FLOAT32_MIN
            
            if 1 == 1:
            
                totcost = np.stack([curcost[top],curcost[bot],curcost[lef],curcost[rit]])
                sort = np.argsort(totcost,axis=0)
                maxcost = sort[3]
                mincost = sort[0]
                ndcost = sort[1]
                rdcost = sort[2]
                
                
                
        
                
                topmax = np.zeros_like(maxcost)
                botmax = np.zeros_like(maxcost)
                lefmax = np.zeros_like(maxcost)
                ritmax = np.zeros_like(maxcost)
                
                topmin = np.zeros_like(mincost)
                botmin = np.zeros_like(mincost)
                lefmin = np.zeros_like(mincost)
                ritmin = np.zeros_like(mincost)
                
                topnd = np.zeros_like(ndcost)
                botnd = np.zeros_like(ndcost)
                lefnd = np.zeros_like(ndcost)
                ritnd = np.zeros_like(ndcost)
                
                toprd = np.zeros_like(rdcost)
                botrd = np.zeros_like(rdcost)
                lefrd = np.zeros_like(rdcost)
                ritrd = np.zeros_like(rdcost)

                topmax[maxcost==0] = FLOAT32_MAX_8RT
                botmax[maxcost==1] = FLOAT32_MAX_8RT
                lefmax[maxcost==2] = FLOAT32_MAX_8RT
                ritmax[maxcost==3] = FLOAT32_MAX_8RT
                
                topmin[mincost==0] = 1
                botmin[mincost==1] = 1
                lefmin[mincost==2] = 1
                ritmin[mincost==3] = 1
                

                topnd[ndcost==0] = 1
                botnd[ndcost==1] = 1
                lefnd[ndcost==2] = 1
                ritnd[ndcost==3] = 1
                
                toprd[rdcost==0] = 1
                botrd[rdcost==1] = 1
                lefrd[rdcost==2] = 1
                ritrd[rdcost==3] = 1
                
                curcost[top] = 0
                curcost[bot] = 0
                curcost[lef] = 0
                curcost[rit] = 0
                
                
                curcost[top] += torch.from_numpy(topmax).float()
                curcost[bot] += torch.from_numpy(botmax).float()
                curcost[lef] += torch.from_numpy(lefmax).float()
                curcost[rit] += torch.from_numpy(ritmax).float()
                
                curcost[top] += torch.from_numpy(topmin).float()
                curcost[bot] += torch.from_numpy(botmin).float()
                curcost[lef] += torch.from_numpy(lefmin).float()
                curcost[rit] += torch.from_numpy(ritmin).float()
                
                curcost[top] += torch.from_numpy(topnd).float()
                curcost[bot] += torch.from_numpy(botnd).float()
                curcost[lef] += torch.from_numpy(lefnd).float()
                curcost[rit] += torch.from_numpy(ritnd).float()
                
                curcost[top] += torch.from_numpy(toprd).float()
                curcost[bot] += torch.from_numpy(botrd).float()
                curcost[lef] += torch.from_numpy(lefrd).float()
                curcost[rit] += torch.from_numpy(ritrd).float()
                
                curcost[top] += (np.random.rand(curcost[top].shape[0])/10).astype(np.float32)
                curcost[bot] += (np.random.rand(curcost[bot].shape[0])/10).astype(np.float32)
                curcost[lef] += (np.random.rand(curcost[lef].shape[0])/10).astype(np.float32)
                curcost[rit] += (np.random.rand(curcost[rit].shape[0])/10).astype(np.float32)

            
            curcost[bord] = FLOAT32_MAX_4RT
            
            rl1 = rf[0,:,:,hj]==2
            rr1 = rf[1,:,:,hj]==2
            rt1 = rf[2,:,:,hj]==2
            rb1 = rf[3,:,:,hj]==2
            
            if torch.sum(rl1)>0:
                curcost[rf[0,:,:,hj]==1,rl1] = FLOAT32_MAX_SQRT
            if torch.sum(rr1)>0: 
                curcost[rf[1,:,:,hj]==1,rr1] = FLOAT32_MAX_SQRT
            if torch.sum(rt1)>0: 
                curcost[rf[2,:,:,hj]==1,rt1] = FLOAT32_MAX_SQRT
            if torch.sum(rb1)>0: 
                curcost[rf[3,:,:,hj]==1, rb1] = FLOAT32_MAX_SQRT
            
    
            curcost = curcost[1:-1,1:-1,1:-1,1:-1]
            curpat = curpat[1:-1,1:-1,1:-1,1:-1]
            
            
            
            curcost = curcost.reshape(curcost.shape[0]*curcost.shape[1],-1).cpu().detach().numpy()
            curpat = curpat.reshape(curpat.shape[0]*curpat.shape[1],-1).cpu().detach().numpy()
            
            
            curcost[curcost==0] = FLOAT32_MAX

            
            row_ind, col_ind = linear_sum_assignment(curcost)
            
            
            
            curpat = curpat[0,:]
            newpat = curpat.copy()
            newpat[row_ind] = curpat[col_ind]
            newpat = torch.from_numpy(newpat).view(patsize+2,patsize+2)
            
    
            
    
            dom2d[0,1:-1,1:-1,hj] = torch.clone(newpat)
        
        base2d = base2d[:,2:-2,2:-2,:]
        dom2d = dom2d[:,2:-2,2:-2,:]
        dummy = dummy[:,2:-2,2:-2,:]
        rf = rf[:,2:-2,2:-2,:]


        
        
        
        

        
        rf = rf[:,1:-1,1:-1,:]
        rfdom = torch.clone(rf)
        

        rf[0,:,:,:] = base2d[0,2:,1:-1,:]
        rf[1,:,:,:]  = base2d[0,:-2,1:-1,:]
        rf[2,:,:,:] = base2d[0,1:-1,2:,:]
        rf[3,:,:,:]  = base2d[0,1:-1,:-2,:]
        
        rfdom[0,:,:,:] = dom2d[0,2:,1:-1,:]
        rfdom[1,:,:,:]  = dom2d[0,:-2,1:-1,:]
        rfdom[2,:,:,:] = dom2d[0,1:-1,2:,:]
        rfdom[3,:,:,:]  = dom2d[0,1:-1,:-2,:]

    
        
        rf = rf.reshape(rf.shape[0],-1,rf.shape[-1])
        rf = torch.abs(rf[0,:,:] - rf[1,:,:]) + torch.abs(rf[0,:,:] - rf[2,:,:])  + torch.abs(rf[0,:,:] - rf[3,:,:]) + torch.abs(rf[1,:,:] - rf[2,:,:]) + torch.abs(rf[1,:,:] - rf[3,:,:]) + torch.abs(rf[2,:,:] - rf[3,:,:])
        rfmean = torch.mean(rf, axis=0).unsqueeze(0)
        rfmean[rfmean==0] = 1
        rf = rf/rfmean
        
        
        rfdom = rfdom.reshape(rfdom.shape[0],-1,rfdom.shape[-1])
        rfdomstd = torch.abs(rfdom[0,:,:] - rfdom[1,:,:]) + torch.abs(rfdom[0,:,:] - rfdom[2,:,:])  + torch.abs(rfdom[0,:,:] - rfdom[3,:,:]) + torch.abs(rfdom[1,:,:] - rfdom[2,:,:]) + torch.abs(rfdom[1,:,:] - rfdom[3,:,:]) + torch.abs(rfdom[2,:,:] - rfdom[3,:,:])
        rfdommean = torch.mean(rfdomstd, axis=0).unsqueeze(0)
        rfdommean[rfdommean==0] = 1
        rfdomstd = rfdomstd/rfdommean
        
        
        
        basediff = torch.abs(rf - rfdomstd)
        
        basediff = torch.sum(basediff, axis=0)

        
        
        
        base2d = base2d.reshape(base2d.shape[0],-1,base2d.shape[-1])
        dom2d = dom2d.reshape(dom2d.shape[0],-1,dom2d.shape[-1])
        dummy = dummy.reshape(dummy.shape[0],-1,dummy.shape[-1])
        
        
        
        trumeanbase = torch.mean(base2d,axis=1)
        trumeanbase = trumeanbase.view(1,1,trumeanbase.shape[-1])
        trumeanbase[trumeanbase==0] = FLOAT32_MIN

        
        
        
        #dom2d = 0*dom2d
        #base2d = 0*base2d
        
        
        base2d = 0*base2d
        dom2d = 0*dom2d



        #counts
        
        
        curcomp = RP(curcomp)
        curcomp = unfold(curcomp)
        curcomp = curcomp.view(-1,patsize+4,patsize+4,curcomp.shape[-1])
        curcomp = curcomp[:,2:-2,2:-2,:]
        
        
        curcomp = curcomp.reshape(1,-1,curcomp.shape[-1])
        
        
        
        dom2d = torch.clone(curcomp)
        
        
        for ow in range(compare3d.shape[1]):
            
            curcomp = compare3d[:,ow:ow+1,:,:]
            curcomp = RP(curcomp)
            curcomp = unfold(curcomp)
            curcomp = curcomp.view(-1,patsize+4,patsize+4,curcomp.shape[-1])
            curcomp = curcomp[:,2:-2,2:-2,:]
            
            rfdom = rfdom.view(4,int(np.sqrt(rfdom.shape[1])),int(np.sqrt(rfdom.shape[1])),-1).float()
            rfdom[0,:,:,:] = curcomp[0,2:,1:-1,:]
            rfdom[1,:,:,:]  = curcomp[0,:-2,1:-1,:]
            rfdom[2,:,:,:] = curcomp[0,1:-1,2:,:]
            rfdom[3,:,:,:]  = curcomp[0,1:-1,:-2,:]
            
  
            
            rfdom = rfdom.reshape(rfdom.shape[0],-1,rfdom.shape[-1])
            rfdomstd = torch.abs(rfdom[0,:,:] - rfdom[1,:,:]) + torch.abs(rfdom[0,:,:] - rfdom[2,:,:])  + torch.abs(rfdom[0,:,:] - rfdom[3,:,:]) + torch.abs(rfdom[1,:,:] - rfdom[2,:,:]) + torch.abs(rfdom[1,:,:] - rfdom[3,:,:]) + torch.abs(rfdom[2,:,:] - rfdom[3,:,:])
            rfdommean = torch.mean(rfdomstd, axis=0).unsqueeze(0)
            rfdommean[rfdommean==0] = 1
            rfdomstd = rfdomstd/rfdommean
            
            
            compdiff = torch.abs(rf - rfdomstd)
        

            compdiff = torch.sum(compdiff, axis=0)
        
            
        
            
            
            
            curcomp = curcomp.reshape(1,-1,curcomp.shape[-1])
            
            
            
            
            trumeancomp = torch.mean(curcomp,axis=1)
            trumeancomp = trumeancomp.view(1,1,trumeancomp.shape[-1])
            trumeancomp[trumeancomp==0] = FLOAT32_MIN
            
            whurdif = torch.where((compdiff<basediff) & (trumeanbase[0,0,:]<trumeancomp[0,0,:]))[0]
            
            
            
            dom2d[:,:,whurdif] = dom2d[:,:,whurdif]*0

            
        
        dom2d = dom2d
        
        
        
        fold = nn.Fold(output_size=(ogshapo[-2],ogshapo[-1]), kernel_size=(patsize, patsize))
        
        dom2d = fold(dom2d)
        dummy = fold(dummy)
        
        dom2d = dom2d/dummy
        
        dom2d = np.clip(dom2d, np.min(inp[oz,:,:]), np.max(inp[oz,:,:]))
        
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"Debleeding was completed in {time.time() - start_time} seconds, starting denoising which usually takes about 40x as long, possibly longer if not using a GPU")
        
        
 

    notdone = True
    learning_rate = 0.00005
    while notdone:
        img = dom2d[0,0,:,:].cpu().detach().numpy()
        
        minner = np.mean(img)
        img = img - minner
        maxer = np.std(img)
        if maxer == 0:
            goodo = False
            notdone = False
            H2 = img + minner
            out[oz,:,:] = H2
            imwrite(file_name[:-4] + '_Channel_'+str(oz+1)+'_Denoised.tif', H2, imagej=True)
            continue
        
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
        

        listimgH = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        
        imgin = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        imgin2 = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        for i in range(imgin.shape[0]):
            for j in range(imgin.shape[1]):
                if j % 2 == 0:
                    imgin[i,j] = imgZ[2*i+1,j]
                    imgin2[i,j] = imgZ[2*i,j]
                if j % 2 == 1:
                    imgin[i,j] = imgZ[2*i,j]
                    imgin2[i,j] = imgZ[2*i+1,j]
        imgin = torch.from_numpy(imgin)
        imgin = torch.unsqueeze(imgin,0)
        imgin = torch.unsqueeze(imgin,0)
        imgin = imgin.to(device)
        imgin2 = torch.from_numpy(imgin2)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = imgin2.to(device)
    
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
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        bothap = imgin
        imgup =  imgin2
        bothapcpu = (bothap[0,0,:,:].cpu().detach().numpy()*maxer+minner)
        imgupcpu =  (imgup[0,0,:,:].cpu().detach().numpy()*maxer+minner)
        imgupcpunow = torch.from_numpy(imgupcpu).view(1,1,imgin.shape[-2],imgin.shape[-1])
        bothapcpunow = torch.from_numpy(bothapcpu).view(1,1,imgin.shape[-2],imgZ.shape[-1])
        
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
            
            
            inputs = tranimg
            inputs = tranimg*maskerto
            labello = tranimg
            
            optimizer.zero_grad()
            outputs = net(inputs,maskerto)
            
            loss1 = torch.mean((1.0-maskerto)*criterion(outputs, labello))
            
            
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
                if np.sum(np.round(H2[1:-1,1:-1]-np.mean(H2[1:-1,1:-1]))>0) <= 25 and learning_rate != 0.000005:
                    goodo = False
                    learning_rate = 0.000005
                    print("Reducing learning rate") 
                    
                last9pct.append(curpct)
                last9pct.pop(0)
                last14img.append(H2)
                last14img.pop(0)
                last10window.append(np.mean(last9pct))
                last10window.pop(0)
                
                if np.argmax(last10window) == 0 and not (last14img[0] is None):
                    goodo = False
                    notdone = False
                    H2 = last14img[0]
                
                keepall[0] = 0*keepall[0]
                keepall[1] = 0*keepall[1]
                keepallcounter = 0
                keepallcounter2 = 0      

                
        imwrite(file_name[:-4] + '_Channel_'+str(oz+1)+'_debleed_denoise.tif', H2, imagej=True)
    
    print("--- %s seconds ---" % (time.time() - start_time)) 
    torch.cuda.empty_cache()
