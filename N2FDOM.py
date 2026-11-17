#Copyright 2025, Jason Lequyer and Laurence Pelletier, All rights reserved.
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
from scipy.sparse import coo_matrix
import scipy.sparse.csgraph as ssc

#N2FDOM

if __name__ == "__main__":
    tsince = 250
    folder = sys.argv[1]
    outfolder = folder+'_N2FDOM'
    Path(outfolder).mkdir(exist_ok=True)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    class TwoCon(nn.Module):
    
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            return x
    
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = TwoCon(1, 64)
            self.conv2 = TwoCon(64, 64)
            self.conv3 = TwoCon(64, 64)
            self.conv4 = TwoCon(64, 64)  
            self.conv6 = nn.Conv2d(64,1,1)
            
    
        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            x = self.conv4(x3)
            x = torch.sigmoid(self.conv6(x))
            return x
        
    file_list = [f for f in os.listdir(folder)]
    start_time = time.time()
    for v in range(len(file_list)):
        
        file_name =  file_list[v]
        print(file_name)
        if file_name[0] == '.':
            continue
        
        img = imread(folder + '/' + file_name)
        noisy = imread(folder.split('_')[0]+'/'+file_name)
        typer = type(img[0,0])
        
        minner = np.amin(img)
        img = img - minner
        maxer = np.amax(img)
        img = img/maxer
        img = img.astype(np.float32)
        shape = img.shape
        
        noisy = noisy-minner
        noisy = noisy/maxer
        
        
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        
        imgZ = imgZ[:,:]
        #imgZ = np.zeros((511,511),dtype=np.float32)
        shape = imgZ.shape
        
        white = np.zeros(imgZ.shape)
        black = np.zeros(imgZ.shape)
        
        start_time = time.time()
        
        for i in range(white.shape[0]):
            for j in range(white.shape[1]):                
                if (i + j) % 2 == 0:
                    black[i,j] = 1
                else:
                    white[i,j] = 1
        
        black = torch.from_numpy(black).to(device)
        white = torch.from_numpy(white).to(device)
                    
                    
        white_crop = white[2:-2,2:-2]
        black_crop = black[2:-2,2:-2]
        
        white = white.view(-1)
        black = black.view(-1)
        
        white_crop = white_crop.reshape(-1)
        black_crop = black_crop.reshape(-1)
        
        white = torch.where(white==1)
        black = torch.where(black==1)
        
        white_crop = torch.where(white_crop==1)
        black_crop = torch.where(black_crop==1)
        
        unfold5 = nn.Unfold(kernel_size=(5, 5))
        
        
        
        paddedimg = torch.from_numpy(imgZ).to(device)
        paddedimg = torch.unsqueeze(paddedimg,0)
        paddedimg = torch.unsqueeze(paddedimg,0)
        
        
        
        counterimg = torch.zeros(imgZ.shape,dtype=torch.float).to(device)
        counterimg = counterimg.view(-1)
        
        increments = torch.arange(counterimg.shape[0]//2).float().to(device)
        
        counterimg[black] = increments
        counterimg[white] = increments
        
        indixes = counterimg.view(1,1,imgZ.shape[0],imgZ.shape[1])
        
        paddedimg = F.pad(paddedimg, (2,2,2,2), "constant",-1)
        indixes = F.pad(indixes, (2,2,2,2), "constant", -1)
        
        paddedimg = unfold5(paddedimg).view(5,5,-1)
        indixes = unfold5(indixes).view(5,5,-1)
        
        left = paddedimg[2,0] + paddedimg[1,1] + paddedimg[3,1]
        right = paddedimg[1,3] + paddedimg[3,3] + paddedimg[2,4]
        down = paddedimg[3,1] + paddedimg[3,3] + paddedimg[4,2]
        up = paddedimg[0,2] + paddedimg[1,1] + paddedimg[1,3]
        curvalo = 3*paddedimg[2,2]
        
        #^Problem b/c of possible negative?
        
        left = (left-curvalo)**2
        right = (right-curvalo)**2
        up = (up-curvalo)**2
        down = (down-curvalo)**2
        
        leftind = indixes[2,1]
        rightind = indixes[2,3]
        downind = indixes[3,2]
        upind = indixes[1,2]
        
        inindixes = indixes[2,2]
        
        left_black = torch.clone(left[black]) 
        right_black = torch.clone(right[black])
        up_black = torch.clone(up[black])
        down_black = torch.clone(down[black])
        
        leftind_black = torch.clone(leftind[black]) 
        rightind_black = torch.clone(rightind[black])
        upind_black = torch.clone(upind[black])
        downind_black = torch.clone(downind[black])
        
        bestos_black = torch.stack([left_black,right_black,down_black,up_black])
        
        prepaddedimg_black = torch.ones_like(bestos_black)
        
        bestos_black = torch.argmax(bestos_black,axis=0)
        
        indos_black = (bestos_black,torch.arange(bestos_black.shape[0]).to(device))
        
        prepaddedimg_black[indos_black] = 100
        
        
        inindixes_black = torch.cat([torch.clone(inindixes[black]),torch.clone(inindixes[black]),torch.clone(inindixes[black]),torch.clone(inindixes[black])])
        paddedimg_black = torch.cat([torch.clone(prepaddedimg_black[0,:]),torch.clone(prepaddedimg_black[1,:]),torch.clone(prepaddedimg_black[2,:]),torch.clone(prepaddedimg_black[3,:])])
        outindixes_black = torch.cat([leftind_black,rightind_black,downind_black,upind_black])
        
        filt_black = torch.where(outindixes_black>-1)
        
        inindixes_black = torch.clone(inindixes_black[filt_black])
        outindixes_black = torch.clone(outindixes_black[filt_black]) 
        paddedimg_black = torch.clone(paddedimg_black[filt_black]) 
        
        paddedimg_black = paddedimg_black.long()
        
        rows_black = inindixes_black.cpu().detach().numpy()
        values_black = paddedimg_black.cpu().detach().numpy()
        cols_black = outindixes_black.cpu().detach().numpy()
        
        coo_black = coo_matrix((values_black, (rows_black, cols_black)), shape=(imgZ.shape[0]*imgZ.shape[1]//2,imgZ.shape[0]*imgZ.shape[1]//2))
        
        match_black = ssc.min_weight_full_bipartite_matching(coo_black)
        
        imgflat_black = torch.from_numpy(imgZ.copy()).to(device).view(-1)
        imgflat_black[black] = imgflat_black[white][match_black[1]]
        
        rows_black = np.concatenate([match_black[0],rows_black])
        cols_black = np.concatenate([match_black[1],cols_black])
        values_black = np.concatenate([[100]*match_black[1].shape[0],values_black])
        
        coo_black = coo_matrix((values_black, (rows_black, cols_black)), shape=(imgZ.shape[0]*imgZ.shape[1]//2,imgZ.shape[0]*imgZ.shape[1]//2))
        
        match_black = ssc.min_weight_full_bipartite_matching(coo_black)

        imgflat_black2 = torch.from_numpy(imgZ.copy()).to(device).view(-1)
        imgflat_black2[black] = imgflat_black2[white][match_black[1]]

        left_white = left[white] 
        right_white = right[white]
        up_white = up[white]
        down_white = down[white]
        
        leftind_white = leftind[white] 
        rightind_white = rightind[white]
        upind_white = upind[white]
        downind_white = downind[white]
        
        bestos_white = torch.stack([left_white,right_white,down_white,up_white])
        
        prepaddedimg_white = torch.ones_like(bestos_white)
        
        bestos_white = torch.argmax(bestos_white,axis=0)
        
        indos_white = (bestos_white,torch.arange(bestos_white.shape[0]).to(device))
        
        prepaddedimg_white[indos_white] = 100
        
        
        inindixes_white = torch.cat([torch.clone(inindixes[white]),torch.clone(inindixes[white]),torch.clone(inindixes[white]),torch.clone(inindixes[white])])
        paddedimg_white = torch.cat([torch.clone(prepaddedimg_white[0,:]),torch.clone(prepaddedimg_white[1,:]),torch.clone(prepaddedimg_white[2,:]),torch.clone(prepaddedimg_white[3,:])])
        outindixes_white = torch.cat([leftind_white,rightind_white,downind_white,upind_white])
        
        filt_white = torch.where(outindixes_white>-1)
        
        inindixes_white = torch.clone(inindixes_white[filt_white])
        outindixes_white = torch.clone(outindixes_white[filt_white]) 
        paddedimg_white = torch.clone(paddedimg_white[filt_white]) 
        
        paddedimg_white = paddedimg_white.long()
        
        rows_white = inindixes_white.cpu().detach().numpy()
        values_white = paddedimg_white.cpu().detach().numpy()
        cols_white = outindixes_white.cpu().detach().numpy()
        
        coo_white = coo_matrix((values_white, (rows_white, cols_white)), shape=(imgZ.shape[0]*imgZ.shape[1]//2,imgZ.shape[0]*imgZ.shape[1]//2))
        
        match_white = ssc.min_weight_full_bipartite_matching(coo_white)
        
        imgflat_white = torch.from_numpy(imgZ.copy()).to(device).view(-1)
        imgflat_white[white] = imgflat_white[black][match_white[1]]
 
        
        rows_white = np.concatenate([match_white[0],rows_white])
        cols_white = np.concatenate([match_white[1],cols_white])
        values_white = np.concatenate([[100]*match_white[1].shape[0],values_white])
        
        coo_white = coo_matrix((values_white, (rows_white, cols_white)), shape=(imgZ.shape[0]*imgZ.shape[1]//2,imgZ.shape[0]*imgZ.shape[1]//2))
        
        match_white = ssc.min_weight_full_bipartite_matching(coo_white)
        
        imgflat_white2 = torch.from_numpy(imgZ.copy()).to(device).view(-1)
        imgflat_white2[white] = imgflat_white2[black][match_white[1]]
 
        both = torch.from_numpy(imgZ.copy()).to(device).view(-1)
        both[white] = imgflat_white[white]
        both[black] = imgflat_black[black]
 
        
        both2 = torch.from_numpy(imgZ.copy()).to(device).view(-1)
        both2[white] = imgflat_white2[white]
        both2[black] = imgflat_black2[black]
        
        
        print("--- %s seconds ---" % (time.time() - start_time))
        
        
        
        imgin5 = imgZ.copy()
        imgin6 = imgZ.copy()
        for i in range(2,imgin5.shape[0]-2):
            for j in range(2,imgin5.shape[1]-2):
                compare = np.array([(imgZ[i-1,j-1]+imgZ[i-1,j+1]+imgZ[i-2,j])/3, (imgZ[i-1,j-1]+imgZ[i+1,j-1]+imgZ[i,j-2])/3, (imgZ[i+1,j+1]+imgZ[i-1,j+1]+imgZ[i,j+2])/3,(imgZ[i+1,j+1]+imgZ[i+1,j-1]+imgZ[i+2,j])/3])
                compare = imgZ[i,j] - compare
                compare = compare**2
                sorto = np.argsort(compare)
                bestcomp = sorto[0]
                sndcomp = sorto[1]
                otcount = []
                otcount2 = []
                if bestcomp == 0:
                    otcount.append(imgZ[i-1,j])  
                if bestcomp == 1:
                    otcount.append(imgZ[i,j-1])
                if bestcomp == 2:
                    otcount.append(imgZ[i,j+1])
                if bestcomp == 3:
                    otcount.append(imgZ[i+1,j])
                if sndcomp == 0:
                    otcount2.append(imgZ[i-1,j])  
                if sndcomp == 1:
                    otcount2.append(imgZ[i,j-1])
                if sndcomp == 2:
                    otcount2.append(imgZ[i,j+1])
                if sndcomp == 3:
                    otcount2.append(imgZ[i+1,j])
                imgin5[i,j] = np.mean(otcount)
                imgin6[i,j] = np.mean(otcount2)

        imgin5 = imgin5[2:-2,2:-2]
        #noise = np.random.normal(0, 21, imgin5.shape)
        #imgin5 = imgin5*maxer+minner+noise
        #imgin5 = imgin5-minner
        
        #imgin5 = imgin5/maxer
        imgin5 = imgin5.astype(np.float32)
        imgin5 = torch.from_numpy(imgin5)
        imgin5 = torch.unsqueeze(imgin5,0)
        imgin5 = torch.unsqueeze(imgin5,0)
        imgin5 = imgin5.to(device)
        
        imginbig5 = torch.clone(imgin5)
        
        
        
        
        
        

        
        
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
        
        listimgV = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        
        imgin3 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        imgin4 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        for i in range(imgin3.shape[0]):
            for j in range(imgin3.shape[1]):
                if i % 2 == 0:
                    imgin3[i,j] = imgZ[i,2*j+1]
                    imgin4[i,j] = imgZ[i, 2*j]
                if i % 2 == 1:
                    imgin3[i,j] = imgZ[i,2*j]
                    imgin4[i,j] = imgZ[i,2*j+1]
        imgin3 = torch.from_numpy(imgin3)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = imgin3.to(device)
        imgin4 = torch.from_numpy(imgin4)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = imgin4.to(device)

        
        
        imgZ = both.view(imgZ.shape[0],imgZ.shape[1]).cpu().detach().numpy()
        
        imgin5 = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        imgin6 = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        for i in range(imgin5.shape[0]):
            for j in range(imgin5.shape[1]):
                if j % 2 == 0:
                    imgin5[i,j] = imgZ[2*i+1,j]
                    imgin6[i,j] = imgZ[2*i,j]
                if j % 2 == 1:
                    imgin5[i,j] = imgZ[2*i,j]
                    imgin6[i,j] = imgZ[2*i+1,j]
        imgin5 = torch.from_numpy(imgin5)
        imgin5 = torch.unsqueeze(imgin5,0)
        imgin5 = torch.unsqueeze(imgin5,0)
        imgin5 = imgin5.to(device)
        imgin6 = torch.from_numpy(imgin6)
        imgin6 = torch.unsqueeze(imgin6,0)
        imgin6 = torch.unsqueeze(imgin6,0)
        imgin6 = imgin6.to(device)
        
        imgin7 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        imgin8 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        for i in range(imgin7.shape[0]):
            for j in range(imgin7.shape[1]):
                if i % 2 == 0:
                    imgin7[i,j] = imgZ[i,2*j+1]
                    imgin8[i,j] = imgZ[i, 2*j]
                if i % 2 == 1:
                    imgin7[i,j] = imgZ[i,2*j]
                    imgin8[i,j] = imgZ[i,2*j+1]
        imgin7 = torch.from_numpy(imgin7)
        imgin7 = torch.unsqueeze(imgin7,0)
        imgin7 = torch.unsqueeze(imgin7,0)
        imgin7 = imgin7.to(device)
        imgin8 = torch.from_numpy(imgin8)
        imgin8 = torch.unsqueeze(imgin8,0)
        imgin8 = torch.unsqueeze(imgin8,0)
        imgin8 = imgin8.to(device)
        
        imgZ = both2.view(imgZ.shape[0],imgZ.shape[1]).cpu().detach().numpy()
        
        imgin9 = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        imgin10 = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        for i in range(imgin9.shape[0]):
            for j in range(imgin9.shape[1]):
                if j % 2 == 0:
                    imgin9[i,j] = imgZ[2*i+1,j]
                    imgin10[i,j] = imgZ[2*i,j]
                if j % 2 == 1:
                    imgin9[i,j] = imgZ[2*i,j]
                    imgin10[i,j] = imgZ[2*i+1,j]

        

        imgin11 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        imgin12 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        for i in range(imgin11.shape[0]):
            for j in range(imgin11.shape[1]):
                if i % 2 == 0:
                    imgin11[i,j] = imgZ[i,2*j+1]
                    imgin12[i,j] = imgZ[i, 2*j]
                if i % 2 == 1:
                    imgin11[i,j] = imgZ[i,2*j]
                    imgin12[i,j] = imgZ[i,2*j+1]

        
        dom1 = imgflat_white.view(imgZ.shape[0],imgZ.shape[1]).cpu().detach().numpy()
        dom2 = imgflat_black.view(imgZ.shape[0],imgZ.shape[1]).cpu().detach().numpy()
        dom3 = imgflat_white2.view(imgZ.shape[0],imgZ.shape[1]).cpu().detach().numpy()
        dom4 = imgflat_black2.view(imgZ.shape[0],imgZ.shape[1]).cpu().detach().numpy()
        
        imgin11 = torch.from_numpy(dom1)
        imgin11 = torch.unsqueeze(imgin11,0)
        imgin11 = torch.unsqueeze(imgin11,0)
        imgin11 = imgin11.to(device)
        imgin12 = torch.from_numpy(dom2)
        imgin12 = torch.unsqueeze(imgin12,0)
        imgin12 = torch.unsqueeze(imgin12,0)
        imgin12 = imgin12.to(device)
        
        imgin9 = torch.from_numpy(dom3)
        imgin9 = torch.unsqueeze(imgin9,0)
        imgin9 = torch.unsqueeze(imgin9,0)
        imgin9 = imgin9.to(device)
        imgin10 = torch.from_numpy(dom4)
        imgin10 = torch.unsqueeze(imgin10,0)
        imgin10 = torch.unsqueeze(imgin10,0)
        imgin10 = imgin10.to(device)
        

        


  
    
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img,0)
        img = torch.unsqueeze(img,0)
        img = img.to(device)
        
        listimgV1 = [[imgin11,imgin12]]
        listimgV2 = [[imgin12,imgin11]]
        listimgV3 = [[imgin9, imgin10]]
        listimgV4 = [[imgin10, imgin9]]
        

        listimgV5 = [[imgin8,imgin4]]
        listimgV6 = [[imgin7,imgin3]]
        listimgV7 = [[imgin6, imgin2]]
        listimgV8 = [[imgin5, imgin]]
        
        listimgV9 = [[imgin, imgin2]]
        listimgV10 = [[imgin2, imgin]]
        listimgV11 = [[imgin3,imgin4]]
        listimgV12 = [[imgin4,imgin3]]
        
        
        
        
        listimgV15 = [[imginbig5, img[:,:,2:-2,2:-2]]]
        
        
        
        
        
        listimg = listimgV1+listimgV2+listimgV3+listimgV4
        #listimg = listimgV1+listimgV2+listimgV3+listimgV4
        
        net = Net()
        net.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        
        
        
        running_loss1=0.0
        running_loss2=0.0
        maxpsnr = -np.inf
        timesince = 0
        last10 = [0]*100
        last10psnr = [0]*100
        cleaned = 0
        counter = 0
        while timesince <= tsince:
            counter += 1
            switcher = np.random.randint(3)
            if switcher == 0:
                masker = torch.rand(imginbig5.shape)
                masker = torch.where(masker>0.6)
                
                inputs = torch.clone(imginbig5)
                labello = torch.clone(img[:,:,2:-2,2:-2])
                temp = torch.clone(inputs[masker])
                inputs[masker] = torch.clone(labello[masker])
                labello[masker] = temp
            else:
                indx = np.random.randint(0,len(listimg))
                data = listimg[indx]
                inputs = data[0]
                labello = data[1]
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss1 = criterion(outputs, labello)
            loss = loss1
            running_loss1+=loss1.item()
            loss.backward()
            optimizer.step()
            
            
            running_loss1=0.0
            with torch.no_grad():
                last10.pop(0)
                last10.append(cleaned*maxer+minner)
                outputstest = net(img)
                cleaned = outputstest[0,0,:,:].cpu().detach().numpy()
                
                noisy = img.cpu().detach().numpy()
                
                ps = -np.mean((noisy-cleaned)**2)
                last10psnr.pop(0)
                last10psnr.append(ps)
                if ps > maxpsnr:
                    maxpsnr = ps
                    outclean = cleaned*maxer+minner
                    timesince = 0
                else:
                    timesince+=1.0
            if counter % 100 == 101:
                H = np.mean(last10, axis=0)       
                imwrite(outfolder + '/' + file_name[:-4] + str(counter) + '.tif', H.astype(typer))
                        
        
        H = np.mean(last10, axis=0)
        
        imwrite(outfolder + '/' + file_name, H.astype(typer))
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        
        
        torch.cuda.empty_cache()
    
