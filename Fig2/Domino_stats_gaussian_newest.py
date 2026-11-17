#Copyright 2022, Jason Lequyer and Laurence Pelletier, All rights reserved.
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
import random






if __name__ == "__main__":
    patsize = 16
    FLOAT32_MIN = 1.175494351E-38
    FLOAT32_MAX = 3.402823466E+38
    FLOAT32_MAX_SQRT = np.sqrt(3.402823466E+38)
    FLOAT32_MAX_4RT = np.sqrt(np.sqrt(3.402823466E+38))
    FLOAT32_MAX_8RT = np.sqrt(np.sqrt(np.sqrt(3.402823466E+38)))
    bstlst = []
    leflst = []
    baselst = []
    randlst = []
    
    lvdomlst = []
    lvbaselst = []
    lvrandlst = []
    lvbeslst = []
    
    diffGTdomlst = []
    diffGTleflst = []
    diffGTbaselst = []
    
    for ia in [15,25,35,50]:
        

        
        
        diff_from_clean = [[],[],[]]
        local_variance = [[],[],[],[]]
    
        noislvl = ia
        
        folder = 'Set12_gaussian'+str(noislvl)
        folder2 = 'Set12_gaussian'+str(noislvl)+'_2'
        clnfolder = 'Set12'
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    
    
        file_list = [f for f in os.listdir(folder) if f[-4:] != '.csv']

        for v in range(len(file_list)):
            
            
            file_name =  file_list[v]
    
            start_time = time.time()
            print(file_name)
            if file_name[0] == '.':
                continue
            
            inp = imread(folder + '/' + file_name)
            inp2 = imread(folder2 + '/' + file_name)
            clninp = imread(clnfolder + '/' + file_name)
            
    
            
            typer = type(inp[0,0])
            
            inpnorm = inp.copy().astype(np.float32)
            inpnorm2 = inp2.copy().astype(np.float32)
            clnnorm = clninp.copy().astype(np.float32)
            
    
            inpnorm = torch.from_numpy(inpnorm)
            inpnorm2 = torch.from_numpy(inpnorm2)
            clnnorm = torch.from_numpy(clnnorm)
            
            
            start_time = time.time()
            
            base2d = torch.clone(inpnorm[:,:])
            cln2d = torch.clone(clnnorm[:,:])
            otherbase = torch.clone(inpnorm2[:,:])
    

            with torch.no_grad():
            
                if len(base2d.shape) == 2:
                    base2d =  torch.unsqueeze(base2d,0)
                base2d =  torch.unsqueeze(base2d,0)
                if len(otherbase.shape) == 2:
                    otherbase =  torch.unsqueeze(otherbase,0)
                otherbase =  torch.unsqueeze(otherbase,0)
                if len(cln2d.shape) == 2:
                    cln2d =  torch.unsqueeze(cln2d,0)
                cln2d =  torch.unsqueeze(cln2d,0)
                
                dummy = torch.ones_like(base2d)
                
            
                
                
                unfold = nn.Unfold(kernel_size=(patsize+4, patsize+4))
                RP = nn.ReflectionPad2d((2, 2, 2, 2))
                base2d = RP(base2d)
                cln2d = RP(cln2d)
                otherbase = RP(otherbase)
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
                cln2d = unfold(cln2d)
                otherbase = unfold(otherbase)
                rf = unfold(rf)
                
                
            
                
                odds = torch.where(odds>0)
                evens = torch.where(evens>0)
            
            
                
                base2d = base2d.view(-1,patsize+4,patsize+4,base2d.shape[-1])
                dummy = dummy.view(-1,patsize+4,patsize+4,dummy.shape[-1])
                otherbase = otherbase.view(-1,patsize+4,patsize+4,otherbase.shape[-1])
                cln2d = cln2d.view(-1,patsize+4,patsize+4,cln2d.shape[-1])
                rf = rf.view(-1,patsize+4,patsize+4,base2d.shape[-1])
                dom2d = torch.clone(base2d)
                dom2dclean = torch.clone(base2d)
                bes2dclean = torch.clone(base2d)
                lef2d = torch.clone(base2d)
                bes2d = torch.clone(base2d)
            
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
                    clnpat = torch.clone(cln2d[0,:,:,hj])
                    
                    
                    curpat = curpat.repeat(patsize+4,patsize+4,1,1)
                    clnpat = clnpat.repeat(patsize+4,patsize+4,1,1)
                    
                
                    
                    
                    curcost = torch.zeros_like(curpat)
                    
                    curcost[top] = torch.abs((2*curpat[top] + curpat[toprit] + curpat[toplef]) - (2*curpat[cn] + curpat[rit] + curpat[lef])) + FLOAT32_MIN
                    curcost[bot] = torch.abs((2*curpat[bot] + curpat[botrit] + curpat[botlef]) - (2*curpat[cn] + curpat[rit] + curpat[lef])) + FLOAT32_MIN
                    curcost[lef] = torch.abs((2*curpat[lef] + curpat[toplef] + curpat[botlef]) - (2*curpat[cn] + curpat[top] + curpat[bot])) + FLOAT32_MIN
                    curcost[rit] = torch.abs((2*curpat[rit] + curpat[toprit] + curpat[botrit]) - (2*curpat[cn] + curpat[top] + curpat[bot])) + FLOAT32_MIN
                    
                    curcostbst = torch.clone(curcost)
                    
                    
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
                    curcostbst[bord] = FLOAT32_MAX_4RT
                    
                    rl1 = rf[0,:,:,hj]==2
                    rr1 = rf[1,:,:,hj]==2
                    rt1 = rf[2,:,:,hj]==2
                    rb1 = rf[3,:,:,hj]==2
                    
                    if torch.sum(rl1)>0:
                        curcost[rf[0,:,:,hj]==1,rl1] = FLOAT32_MAX_SQRT
                        curcostbst[rf[0,:,:,hj]==1,rl1] = FLOAT32_MAX_SQRT
                    if torch.sum(rr1)>0: 
                        curcost[rf[1,:,:,hj]==1,rr1] = FLOAT32_MAX_SQRT
                        curcostbst[rf[1,:,:,hj]==1,rr1] = FLOAT32_MAX_SQRT
                    if torch.sum(rt1)>0: 
                        curcost[rf[2,:,:,hj]==1,rt1] = FLOAT32_MAX_SQRT
                        curcostbst[rf[2,:,:,hj]==1,rt1] = FLOAT32_MAX_SQRT
                    if torch.sum(rb1)>0: 
                        curcost[rf[3,:,:,hj]==1, rb1] = FLOAT32_MAX_SQRT
                        curcostbst[rf[3,:,:,hj]==1, rb1] = FLOAT32_MAX_SQRT
                    
                    
                    curcost = curcost[1:-1,1:-1,1:-1,1:-1]
                    curcostbst = curcostbst[1:-1,1:-1,1:-1,1:-1]

                
                    curpat = curpat[1:-1,1:-1,1:-1,1:-1]
                    clnpat = clnpat[1:-1,1:-1,1:-1,1:-1]
                    
                    
                    
                    curcost = curcost.reshape(curcost.shape[0]*curcost.shape[1],-1).cpu().detach().numpy()
                    curcostbst = curcostbst.reshape(curcostbst.shape[0]*curcostbst.shape[1],-1).cpu().detach().numpy()
                    curpat = curpat.reshape(curpat.shape[0]*curpat.shape[1],-1).cpu().detach().numpy()
                    clnpat = clnpat.reshape(clnpat.shape[0]*clnpat.shape[1],-1).cpu().detach().numpy()
                    
                    
                    curcost[curcost==0] = FLOAT32_MAX
                    curcostbst[curcostbst==0] = FLOAT32_MAX

                    
                    row_ind, col_ind = linear_sum_assignment(curcost)
                    
                    bst_nay = np.argsort(curcostbst, axis=1)[:,0]
                    
                    curpat = curpat[0,:]
                    clnpat = clnpat[0,:]
                    newpat = curpat.copy()
                    newpatcln = clnpat.copy()
                    bstpat = curpat.copy()
                    bstpatcln = clnpat.copy()
                    bstpat[row_ind] = curpat[bst_nay]
                    bstpatcln[row_ind] = clnpat[bst_nay]
                    newpat[row_ind] = curpat[col_ind]
                    newpatcln[row_ind] = clnpat[col_ind]
                    newpat = torch.from_numpy(newpat).view(patsize+2,patsize+2)
                    bstpat = torch.from_numpy(bstpat).view(patsize+2,patsize+2)
                    bstpatcln = torch.from_numpy(bstpatcln).view(patsize+2,patsize+2)
                    curpat = torch.from_numpy(curpat).view(patsize+2,patsize+2)
                    clnpat = torch.from_numpy(clnpat).view(patsize+2,patsize+2)
                    newpatcln = torch.from_numpy(newpatcln).view(patsize+2,patsize+2)

                        
            
                    dom2d[0,1:-1,1:-1,hj] = torch.clone(newpat)
                    dom2dclean[0,1:-1,1:-1,hj] = torch.clone(newpatcln)
                    bes2dclean[0,1:-1,1:-1,hj] = torch.clone(bstpatcln)
                    bes2d[0,1:-1,1:-1,hj] = torch.clone(bstpat)
                    lef2d[0,0:-2,1:-1,hj] = torch.clone(curpat)
                
                base2d = base2d[:,2:-2,2:-2,:]
                dom2d = dom2d[:,2:-2,2:-2,:]
                dom2dclean = dom2dclean[:,2:-2,2:-2,:]
                bes2dclean = bes2dclean[:,2:-2,2:-2,:]
                bes2d = bes2d[:,2:-2,2:-2,:]
                lef2d = lef2d[:,2:-2,2:-2,:]
                dummy = dummy[:,2:-2,2:-2,:]
                otherbase = otherbase[:,2:-2,2:-2,:]
                cln2d = cln2d[:,2:-2,2:-2,:]
                rf = rf[:,2:-2,2:-2,:]
                
                rand2d = torch.clone(base2d)
                rand2dclean = torch.clone(cln2d)
                for ia in range(1,rand2d.shape[1]-1):
                    for ja in range(1,rand2d.shape[2]-1):
                        for ka in range(rand2d.shape[3]):
                            ido = np.random.randint(4)
                            if ido==0:
                                rand2d[0,ia,ja,ka] = base2d[0,ia-1,ja,ka]
                                rand2dclean[0,ia,ja,ka] = cln2d[0,ia-1,ja,ka]
                            elif ido==1:
                                rand2d[0,ia,ja,ka] = base2d[0,ia+1,ja,ka]
                                rand2dclean[0,ia,ja,ka] = cln2d[0,ia+1,ja,ka]
                            elif ido==2:
                                rand2d[0,ia,ja,ka] = base2d[0,ia,ja-1,ka]
                                rand2dclean[0,ia,ja,ka] = cln2d[0,ia,ja-1,ka]
                            else:
                                rand2d[0,ia,ja,ka] = base2d[0,ia,ja+1,ka]
                                rand2dclean[0,ia,ja,ka] = cln2d[0,ia,ja+1,ka]
                

                
                rf = rf[:,1:-1,1:-1,:]
                rfdom = torch.clone(rf)
                rfoth = torch.clone(rf)
                rfcln = torch.clone(rf)
                rflef = torch.clone(rf)
                rfrand = torch.clone(rf)
                rfbes = torch.clone(rf)
                rfbescln = torch.clone(rf)
                rfdomcln = torch.clone(rf)
                rfrandcln = torch.clone(rf)
                

                rf[0,:,:,:] = base2d[0,2:,1:-1,:]
                rf[1,:,:,:]  = base2d[0,:-2,1:-1,:]
                rf[2,:,:,:] = base2d[0,1:-1,2:,:]
                rf[3,:,:,:]  = base2d[0,1:-1,:-2,:]
                
 
                
                rfoth[0,:,:,:] = otherbase[0,2:,1:-1,:]
                rfoth[1,:,:,:]  = otherbase[0,:-2,1:-1,:]
                rfoth[2,:,:,:] = otherbase[0,1:-1,2:,:]
                rfoth[3,:,:,:]  = otherbase[0,1:-1,:-2,:]
                
                rfcln[0,:,:,:] = cln2d[0,2:,1:-1,:]
                rfcln[1,:,:,:]  = cln2d[0,:-2,1:-1,:]
                rfcln[2,:,:,:] = cln2d[0,1:-1,2:,:]
                rfcln[3,:,:,:]  = cln2d[0,1:-1,:-2,:]
                
                rfbes[0,:,:,:] = bes2d[0,2:,1:-1,:]
                rfbes[1,:,:,:]  = bes2d[0,:-2,1:-1,:]
                rfbes[2,:,:,:] = bes2d[0,1:-1,2:,:]
                rfbes[3,:,:,:]  = bes2d[0,1:-1,:-2,:]
                
                rfbescln[0,:,:,:] = bes2dclean[0,2:,1:-1,:]
                rfbescln[1,:,:,:]  = bes2dclean[0,:-2,1:-1,:]
                rfbescln[2,:,:,:] = bes2dclean[0,1:-1,2:,:]
                rfbescln[3,:,:,:]  = bes2dclean[0,1:-1,:-2,:]
                
                rfdom[0,:,:,:] = dom2d[0,2:,1:-1,:]
                rfdom[1,:,:,:]  = dom2d[0,:-2,1:-1,:]
                rfdom[2,:,:,:] = dom2d[0,1:-1,2:,:]
                rfdom[3,:,:,:]  = dom2d[0,1:-1,:-2,:]
                
                rflef[0,:,:,:] = lef2d[0,2:,1:-1,:]
                rflef[1,:,:,:]  = lef2d[0,:-2,1:-1,:]
                rflef[2,:,:,:] = lef2d[0,1:-1,2:,:]
                rflef[3,:,:,:]  = lef2d[0,1:-1,:-2,:]
                
                rfrand[0,:,:,:] = rand2d[0,2:,1:-1,:]
                rfrand[1,:,:,:]  = rand2d[0,:-2,1:-1,:]
                rfrand[2,:,:,:] = rand2d[0,1:-1,2:,:]
                rfrand[3,:,:,:]  = rand2d[0,1:-1,:-2,:]
                
                
                rfrandcln[0,:,:,:] = rand2dclean[0,2:,1:-1,:]
                rfrandcln[1,:,:,:]  = rand2dclean[0,:-2,1:-1,:]
                rfrandcln[2,:,:,:] = rand2dclean[0,1:-1,2:,:]
                rfrandcln[3,:,:,:]  = rand2dclean[0,1:-1,:-2,:]
                
                rfdomcln[0,:,:,:] = dom2dclean[0,2:,1:-1,:]
                rfdomcln[1,:,:,:]  = dom2dclean[0,:-2,1:-1,:]
                rfdomcln[2,:,:,:] = dom2dclean[0,1:-1,2:,:]
                rfdomcln[3,:,:,:]  = dom2dclean[0,1:-1,:-2,:]
                
                
                n2nvGT = torch.log(torch.mean((base2d- cln2d)**2))
                lefvGT = torch.log(torch.mean((lef2d - cln2d)**2))
                domvGT = torch.log(torch.mean((dom2d - cln2d)**2))
                
                diffGTleflst.append(lefvGT.item())
                diffGTbaselst.append(n2nvGT.item())
                diffGTdomlst.append(domvGT.item())
                

                rfcln = rfcln.reshape(rfcln.shape[0],-1,rfcln.shape[-1])
                rfclnstd = torch.abs(rfcln[0,:,:] - rfcln[1,:,:]) + torch.abs(rfcln[0,:,:] - rfcln[2,:,:])  + torch.abs(rfcln[0,:,:] - rfcln[3,:,:]) + torch.abs(rfcln[1,:,:] - rfcln[2,:,:]) + torch.abs(rfcln[1,:,:] - rfcln[3,:,:]) + torch.abs(rfcln[2,:,:] - rfcln[3,:,:])
                rfclnmean = torch.mean(rfclnstd, axis=0).unsqueeze(0)
                rfclnmean[rfclnmean==0] = 1
                rfclnstd = rfclnstd/rfclnmean                
                
                rf = rf.reshape(rf.shape[0],-1,rf.shape[-1])
                rfcln = rfcln.reshape(rfcln.shape[0],-1,rfcln.shape[-1])
                baselv = torch.mean(torch.var(rf-rfcln,axis=0)).item()

                
                
                rfdom = rfdom.reshape(rfdom.shape[0],-1,rfdom.shape[-1])
                rfdomcln = rfdomcln.reshape(rfdomcln.shape[0],-1,rfdomcln.shape[-1])
                domlv = torch.mean(torch.var(rfdom-rfdomcln,axis=0)).item()
                rfdomstd = torch.abs(rfdom[0,:,:] - rfdom[1,:,:]) + torch.abs(rfdom[0,:,:] - rfdom[2,:,:])  + torch.abs(rfdom[0,:,:] - rfdom[3,:,:]) + torch.abs(rfdom[1,:,:] - rfdom[2,:,:]) + torch.abs(rfdom[1,:,:] - rfdom[3,:,:]) + torch.abs(rfdom[2,:,:] - rfdom[3,:,:])
                
                rfdommean = torch.mean(rfdomstd, axis=0).unsqueeze(0)
                rfdommean[rfdommean==0] = 1
                rfdomstd = rfdomstd/rfdommean
                

                
                rfoth = rfoth.reshape(rfoth.shape[0],-1,rfoth.shape[-1])
                rfothstd = torch.abs(rfoth[0,:,:] - rfoth[1,:,:]) + torch.abs(rfoth[0,:,:] - rfoth[2,:,:])  + torch.abs(rfoth[0,:,:] - rfoth[3,:,:]) + torch.abs(rfoth[1,:,:] - rfoth[2,:,:]) + torch.abs(rfoth[1,:,:] - rfoth[3,:,:]) + torch.abs(rfoth[2,:,:] - rfoth[3,:,:])
                rfothmean = torch.mean(rfothstd, axis=0).unsqueeze(0)
                rfothmean[rfothmean==0] = 1
                rfothstd = rfothstd/rfothmean
                        
                rflef = rflef.reshape(rflef.shape[0],-1,rflef.shape[-1])
                rflefstd = torch.abs(rflef[0,:,:] - rflef[1,:,:]) + torch.abs(rflef[0,:,:] - rflef[2,:,:])  + torch.abs(rflef[0,:,:] - rflef[3,:,:]) + torch.abs(rflef[1,:,:] - rflef[2,:,:]) + torch.abs(rflef[1,:,:] - rflef[3,:,:]) + torch.abs(rflef[2,:,:] - rflef[3,:,:])
                rflefmean = torch.mean(rflefstd, axis=0).unsqueeze(0)
                rflefmean[rflefmean==0] = 1
                rflefstd = rflefstd/rflefmean
                
                rfrand = rfrand.reshape(rfrand.shape[0],-1,rfrand.shape[-1])
                rfrandcln = rfrandcln.reshape(rfrandcln.shape[0],-1,rfrandcln.shape[-1])
                randlv = torch.mean(torch.var(rfrand-rfrandcln,axis=0)).item()
                rfrandstd = torch.abs(rfrand[0,:,:] - rfrand[1,:,:]) + torch.abs(rfrand[0,:,:] - rfrand[2,:,:])  + torch.abs(rfrand[0,:,:] - rfrand[3,:,:]) + torch.abs(rfrand[1,:,:] - rfrand[2,:,:]) + torch.abs(rfrand[1,:,:] - rfrand[3,:,:]) + torch.abs(rfrand[2,:,:] - rfrand[3,:,:])

                rfrandmean = torch.mean(rfrandstd, axis=0).unsqueeze(0)
                rfrandmean[rfrandmean==0] = 1
                rfrandstd = rfrandstd/rfrandmean

                
                rf = torch.abs(rf[0,:,:] - rf[1,:,:]) + torch.abs(rf[0,:,:] - rf[2,:,:])  + torch.abs(rf[0,:,:] - rf[3,:,:]) + torch.abs(rf[1,:,:] - rf[2,:,:]) + torch.abs(rf[1,:,:] - rf[3,:,:]) + torch.abs(rf[2,:,:] - rf[3,:,:])
                
                rfmean = torch.mean(rf, axis=0).unsqueeze(0)
                rfmean[rfmean==0] = 1
                rf = rf/rfmean
                
                rfbes = rfbes.reshape(rfbes.shape[0],-1,rfbes.shape[-1])
                rfbescln = rfbescln.reshape(rfbescln.shape[0],-1,rfbescln.shape[-1])
                beslv = torch.mean(torch.var(rfbes-rfbescln,axis=0)).item()
                
                lvdomlst.append(domlv)
                lvbaselst.append(baselv)
                lvrandlst.append(randlv)
                lvbeslst.append(beslv)
                
                
                

                rfbesstd = torch.abs(rfbes[0,:,:] - rfbes[1,:,:]) + torch.abs(rfbes[0,:,:] - rfbes[2,:,:])  + torch.abs(rfbes[0,:,:] - rfbes[3,:,:]) + torch.abs(rfbes[1,:,:] - rfbes[2,:,:]) + torch.abs(rfbes[1,:,:] - rfbes[3,:,:]) + torch.abs(rfbes[2,:,:] - rfbes[3,:,:])
                rfbesmean = torch.mean(rfbesstd, axis=0).unsqueeze(0)
                rfbesmean[rfbesmean==0] = 1
                rfbesstd = rfbesstd/rfbesmean
                
                basediff = torch.abs(rf - rfdomstd)
                
                n2ndiff = torch.abs(rf - rfothstd)
                
                bstdiff = torch.abs(rf - rfbesstd)
                
                lefdiff = torch.abs(rf - rflefstd)
                
                randdiff = torch.abs(rf - rfrandstd)
                
                basediff = torch.sum(basediff, axis=0)
                
                n2ndiff = torch.sum(n2ndiff, axis=0)
                
                bstdiff = torch.sum(bstdiff, axis=0)
                
                lefdiff = torch.sum(lefdiff, axis=0)
                
                randdiff = torch.sum(randdiff, axis=0)
                
                bstlst.append(torch.log(torch.mean((bstdiff - n2ndiff)**2)).item())
                leflst.append(torch.log(torch.mean((lefdiff - n2ndiff)**2)).item())
                baselst.append(torch.log(torch.mean((basediff - n2ndiff)**2)).item())
                randlst.append(torch.log(torch.mean((randdiff - n2ndiff)**2)).item())
                
                #print([baselv,domlv,randlv])
                #print([n2nvGT,domvGT,lefvGT])

    bstlst = list(torch.tensor(bstlst).cpu().detach().numpy())
    leflst = list(torch.tensor(leflst).cpu().detach().numpy())
    baselst = list(torch.tensor(baselst).cpu().detach().numpy())
    randlst = list(torch.tensor(randlst).cpu().detach().numpy())
    
    bstlst = list((np.array(bstlst)[:12]+np.array(bstlst)[12:24]+np.array(bstlst)[24:36]+np.array(bstlst)[36:48])/4)
    leflst = list((np.array(leflst)[:12]+np.array(leflst)[12:24]+np.array(leflst)[24:36]+np.array(leflst)[36:48])/4)
    baselst = list((np.array(baselst)[:12]+np.array(baselst)[12:24]+np.array(baselst)[24:36]+np.array(baselst)[36:48])/4)
    randlst = list((np.array(randlst)[:12]+np.array(randlst)[12:24]+np.array(randlst)[24:36]+np.array(randlst)[36:48])/4)
    
    lvdomlst = list((np.array(lvdomlst)[:12]+np.array(lvdomlst)[12:24]+np.array(lvdomlst)[24:36]+np.array(lvdomlst)[36:48])/4)
    lvbaselst = list((np.array(lvbaselst)[:12]+np.array(lvbaselst)[12:24]+np.array(lvbaselst)[24:36]+np.array(lvbaselst)[36:48])/4)
    lvrandlst = list((np.array(lvrandlst)[:12]+np.array(lvrandlst)[12:24]+np.array(lvrandlst)[24:36]+np.array(lvrandlst)[36:48])/4)
    lvbeslst = list((np.array(lvbeslst)[:12]+np.array(lvbeslst)[12:24]+np.array(lvbeslst)[24:36]+np.array(lvbeslst)[36:48])/4)
    
    diffGTdomlst = list((np.array(diffGTdomlst)[:12]+np.array(diffGTdomlst)[12:24]+np.array(diffGTdomlst)[24:36]+np.array(diffGTdomlst)[36:48])/4)
    diffGTbaselst = list((np.array(diffGTbaselst)[:12]+np.array(diffGTbaselst)[12:24]+np.array(diffGTbaselst)[24:36]+np.array(diffGTbaselst)[36:48])/4)
    diffGTleflst = list((np.array(diffGTleflst)[:12]+np.array(diffGTleflst)[12:24]+np.array(diffGTleflst)[24:36]+np.array(diffGTleflst)[36:48])/4)
    
    df = pd.DataFrame(zip(*[baselst,bstlst,randlst,leflst])).astype("float")
    df.columns = ['DOMINO','BESTSHIFT','RANDOMSHIFT','LEFTSHIFT']
    df.to_csv('LogDifferenceWithN2N.csv')
    
    df = pd.DataFrame(zip(*[lvrandlst,lvbeslst,lvbaselst,lvdomlst])).astype("float")
    df.columns = ['RANDSHIFT','BESTSHIFT','ORIGINAL','DOMINO']
    df.to_csv('LocalVariance.csv')
    
    df = pd.DataFrame(zip(*[diffGTleflst,diffGTbaselst,diffGTdomlst])).astype("float")
    df.columns = ['LEFTSHIFT','ORIGINAL','DOMINO']
    df.to_csv('DifferenceFromGT.csv')
    
    
    sys.exit()
    bstlstlst.append(bstlst)   
    baselstlst.append(baselst)   
    randlstlst.append(randlst)   
    leflstlst.append(leflst)  
    
    

    
    df = pd.DataFrame(zip(*[baselstlst[1],bstlstlst[1],randlstlst[1],leflstlst[1]])).astype("float")
    df.columns = ['|DOMINO_cutoff - N2N_cutoff|', '|BESTSHIFT_cutoff - N2N_cutoff|','|RANDOMSHIFT_cutoff - N2N_cutoff|','|LEFTSHIFT_cutoff - N2N_cutoff|']
    df.to_csv('Gaussian25.csv')
    
    df = pd.DataFrame(zip(*[baselstlst[2],bstlstlst[2],randlstlst[2],leflstlst[2]])).astype("float")
    df.columns = ['|DOMINO_cutoff - N2N_cutoff|', '|BESTSHIFT_cutoff - N2N_cutoff|','|RANDOMSHIFT_cutoff - N2N_cutoff|','|LEFTSHIFT_cutoff - N2N_cutoff|']
    df.to_csv('Gaussian35.csv')
    
    df = pd.DataFrame(zip(*[baselstlst[3],bstlstlst[3],randlstlst[3],leflstlst[3]])).astype("float")
    df.columns = ['|DOMINO_cutoff - N2N_cutoff|', '|BESTSHIFT_cutoff - N2N_cutoff|','|RANDOMSHIFT_cutoff - N2N_cutoff|','|LEFTSHIFT_cutoff - N2N_cutoff|']
    df.to_csv('Gaussian50.csv')
            
    df = pd.DataFrame(local_variance).transpose()
    df.columns = ['Noisy', 'Random Neighbour', 'Avg Neighbour', 'Domino']
    df.to_csv('Gaussian'+str(noislvl)+'_localvariance.csv')
       
            
