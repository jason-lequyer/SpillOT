from tifffile import imread
import os
import sys
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
#compute_psnr_ssim

if __name__ == "__main__":
    
    GTdir = sys.argv[1]
    noisydir = sys.argv[2]
    drange = int(sys.argv[3])
    file_list = [f for f in os.listdir(noisydir)]
    numberofpoints = len(file_list)
    
    counter = 0
    avgps = 0
    avgss = 0
    for v in range(numberofpoints):
        filename = file_list[v]
        if filename[0] == '.':
            continue
        print(filename)
        counter += 1
        img = imread(GTdir + '/' + filename)
        GT = imread(noisydir + '/' + filename)
        ps = psnr(GT,img,data_range = drange)
        ss = ssim(GT, img, gaussian_weights = True, sigma=1.5, use_sample_covariance=False, data_range = drange)
        avgps += ps
        avgss += ss
        print(ps)
    avgps = avgps/counter
    avgss = avgss/counter
    print('PSNR: '+str(avgps))
    print('SSIM: '+str(avgss))
