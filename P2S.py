import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from pathlib import Path
import os
from tifffile import imread, imwrite
import argparse
from collections import namedtuple
#P2s

folder = sys.argv[1]
outfolder = folder+'_P2S'
Path(outfolder).mkdir(exist_ok=True)

ListaParams = namedtuple('ListaParams', ['kernel_size', 'num_filters', 'stride', 'unfoldings', 'channels'])
ListaParams = namedtuple('ListaParams', ['kernel_size', 'num_filters', 'stride', 'unfoldings', 'channels'])
def calc_pad_sizes(I: torch.Tensor, kernel_size: int, stride: int):
    left_pad = stride
    right_pad = 0 if (I.shape[3] + left_pad - kernel_size) % stride == 0 else stride - ((I.shape[3] + left_pad - kernel_size) % stride)
    top_pad = stride
    bot_pad = 0 if (I.shape[2] + top_pad - kernel_size) % stride == 0 else stride - ((I.shape[2] + top_pad - kernel_size) % stride)
    right_pad += stride
    bot_pad += stride
    return left_pad, right_pad, top_pad, bot_pad

class SoftThreshold(nn.Module):
    def __init__(self, size, init_threshold=1e-3):
        super(SoftThreshold, self).__init__()
        self.threshold = nn.Parameter(init_threshold * torch.ones(1,size,1,1))

    def forward(self, x):
        mask1 = (x > self.threshold).float()
        mask2 = (x < -self.threshold).float()
        out = mask1.float() * (x - self.threshold)
        out += mask2.float() * (x + self.threshold)
        return out





class ConvLista_T(nn.Module):
    def __init__(self, params: ListaParams, A=None, B=None, C=None, threshold=1e-2, norm=False):
        super(ConvLista_T, self).__init__()
        if A is None:
            A = torch.randn(params.num_filters, params.channels, params.kernel_size, params.kernel_size)
            l = conv_power_method(A, [128, 128], num_iters=20, stride=params.stride)
            # l = conv_power_method(A, [28,28], num_iters=200, stride=params.stride)
            A /= torch.sqrt(l)
        if B is None:
            B = torch.clone(A)
        if C is None:
            C = torch.clone(A)
        self.apply_A = torch.nn.ConvTranspose2d(params.num_filters, params.channels, kernel_size=params.kernel_size,
                                                stride=params.stride, bias=False)
        self.apply_B = torch.nn.Conv2d(params.channels, params.num_filters, kernel_size=params.kernel_size, stride=params.stride, bias=False)
        self.apply_C = torch.nn.ConvTranspose2d(params.num_filters, params.channels, kernel_size=params.kernel_size,
                                                stride=params.stride, bias=False)
        self.apply_A.weight.data = A
        self.apply_B.weight.data = B
        self.apply_C.weight.data = C
        self.soft_threshold = SoftThreshold(params.num_filters, threshold)
        self.params = params
        self.num_iter = params.unfoldings
        # self.norm = norm
        # if self.norm:
            # self.norm_layer = torch.nn.InstanceNorm2d(params.num_filters)
            # self.norm_layer = torch.nn.

    def _split_image(self, I):
        if self.params.stride == 1:
            return I, torch.ones_like(I)
        left_pad, right_pad, top_pad, bot_pad = calc_pad_sizes(I, self.params.kernel_size, self.params.stride)
        I_batched_padded = torch.zeros(I.shape[0], self.params.stride ** 2, I.shape[1], top_pad + I.shape[2] + bot_pad,
                                       left_pad + I.shape[3] + right_pad).type_as(I)
        valids_batched = torch.zeros_like(I_batched_padded)
        for num, (row_shift, col_shift) in enumerate([(i, j) for i in range(self.params.stride) for j in range(self.params.stride)]):
            I_padded = F.pad(I, pad=(
            left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='reflect')
            valids = F.pad(torch.ones_like(I), pad=(
            left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='constant')
            I_batched_padded[:, num, :, :, :] = I_padded
            valids_batched[:, num, :, :, :] = valids
        I_batched_padded = I_batched_padded.reshape(-1, *I_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return I_batched_padded, valids_batched
    
    def disable_warmup(self):
        self.num_iter = self.params.unfoldings
        
    def enable_warmup(self):
        self.num_iter = 1
    
    def forward(self, I):
        I_batched_padded, valids_batched = self._split_image(I)
        conv_input = self.apply_B(I_batched_padded) #encode
        gamma_k = self.soft_threshold(conv_input)
        # ic(gamma_k.shape)
        for k in range(self.num_iter - 1):
            x_k = self.apply_A(gamma_k) # decode
            # r_k = self.apply_B(x_k-I_batched_padded) #encode
            r_k = self.apply_B(x_k-I_batched_padded) #encode
            # if self.norm:
                # r_k = self.norm_layer(r_k)
                #bug? try adding
            gamma_k = self.soft_threshold(gamma_k - r_k)
        output_all = self.apply_C(gamma_k)
        output_cropped = torch.masked_select(output_all, valids_batched.bool()).reshape(I.shape[0], self.params.stride ** 2, *I.shape[1:])
        # if self.return_all:
        #     return output_cropped
        output = output_cropped.mean(dim=1, keepdim=False)
        # output = F.relu(output)
        return torch.clamp(output,0.0,1.0)
    
def conv_power_method(D, image_size, num_iters=100, stride=1):
    """
    Finds the maximal eigenvalue of D.T.dot(D) using the iterative power method
    :param D:
    :param num_needles:
    :param image_size:
    :param patch_size:
    :param num_iters:
    :return:
    """
    needles_shape = [int(((image_size[0] - D.shape[-2])/stride)+1), int(((image_size[1] - D.shape[-1])/stride)+1)]
    x = torch.randn(1, D.shape[0], *needles_shape).type_as(D)
    for _ in range(num_iters):
        c = torch.norm(x.reshape(-1))
        x = x / c
        y = F.conv_transpose2d(x, D, stride=stride)
        x = F.conv2d(y, D, stride=stride)
    return torch.norm(x.reshape(-1))

    
def build_model(cfg):
    params = ListaParams(cfg['model_cfg']['kernel_size'], cfg['model_cfg']['num_filters'], cfg['model_cfg']['stride'], 
        cfg['model_cfg']['num_iter'], cfg['model_cfg']['channels'])
    net = ConvLista_T(params, threshold=cfg['model_cfg']['threshold'], norm=cfg['model_cfg']['norm'])
    return net
    

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)
operation_seed_counter = 0
def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2

def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        # ic(img_per_channel.shape, subimage.shape)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        channel_mask = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
        # ic(channel_mask.shape, subimage.shape)
        subimage[:, i:i+1, :, :] = channel_mask
    return subimage

file_list = [f for f in os.listdir(folder)]
for v in range(len(file_list)):
    print(file_list[v])
    input_path = folder+'/'+file_list[v]
    output_path = outfolder+'/'+file_list[v]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=input_path)
    parser.add_argument("--output_path", default=output_path)



    
    
    
    
    def main(noisy, config, experiment_cfg):
        model = build_model(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(
            "Number of params: ",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        # optimizer
        if experiment_cfg["optimizer"] == "Adam":
            LR = experiment_cfg["lr"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
        # psnr_list = []
        # loss_list = []
        # ssims_list = []
        exp_weight = 0.99
    
        out_avg = None
    
        noisy_in = noisy
        noisy_in = noisy_in.to(device)
    
        H = None
        W = None
        # if noisy.shape[1] != noisy.shape[2]:
        #     H = noisy.shape[2]
        #     W = noisy.shape[3]
        #     val_size = (max(H, W) + 31) // 32 * 32
        #     noisy_in = TF.pad(
        #         noisy,
        #         (0, 0, val_size - noisy.shape[3], val_size - noisy.shape[2]),
        #         padding_mode="reflect",
        #     )
        
        t = range(experiment_cfg["num_iter"])
        pll = nn.PoissonNLLLoss(log_input=False, full=True)
        last_net = None
        psrn_noisy_last = 0.0
        for i in t:
    
            mask1, mask2 = generate_mask_pair(noisy_in)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)
            with torch.no_grad():
                noisy_denoised = model(noisy_in)
                noisy_denoised = torch.clamp(noisy_denoised, 0.0, 1.0)
    
            noisy_in_aug = noisy_in.clone()
            # ic(noisy_in_aug.shape, mask1.shape, noisy_in.shape)
            noisy_sub1 = generate_subimages(noisy_in_aug, mask1)
            noisy_sub2 = generate_subimages(noisy_in_aug, mask2)
    
            noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
            noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)
    
            noisy_output = model(noisy_sub1)
            noisy_output = torch.clamp(noisy_output, 0.0, 1.0)
            noisy_target = noisy_sub2
    
            Lambda = experiment_cfg["LAM"]
            diff = noisy_output - noisy_target
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
    
            if "l1" in experiment_cfg.keys():
                l1_regularization = 0.0
                for param in model.parameters():
                    l1_regularization += param.abs().sum()
                total_loss = experiment_cfg["l1"] * l1_regularization
            # else:
            if "poisson_loss" in experiment_cfg.keys():
                loss1 = pll(noisy_output, noisy_target)
                loss2 = F.l1_loss(noisy_output, noisy_target)
                loss1 += loss2
            elif "poisson_loss_only" in experiment_cfg.keys():
                loss1 = pll(noisy_output, noisy_target)
            elif "l1_loss" in experiment_cfg.keys():
                loss1 = F.l1_loss(noisy_output, noisy_target)
    
            elif "mse" in experiment_cfg.keys():
                loss1 = torch.mean(diff ** 2)
            else:
                loss1 = F.l1_loss(noisy_output, noisy_target)
                # orch.mean(diff**2)
            loss2 = Lambda * torch.mean((diff - exp_diff) ** 2)
    
            loss = loss1 + loss2
            if "l1" in experiment_cfg.keys():
                loss += total_loss
            loss.backward()
    
            with torch.no_grad():
                out_full = model(noisy_in).detach().cpu()
                if H is not None:
                    out_full = out_full[:, :, :H, :W]
                if out_avg is None:
                    out_avg = out_full.detach().cpu()
                else:
                    out_avg = out_avg * exp_weight + out_full * (1 - exp_weight)
                    out_avg = out_avg.detach().cpu()
                noisy_psnr = torch.mean((out_full - noisy_in.detach().cpu())**2)
    
            if (i + 1) % 50:
                if 10*np.log10(noisy_psnr/psrn_noisy_last) < -4 and last_net is not None:
                    print("Falling back to previous checkpoint.")
    
                    for new_param, net_param in zip(last_net, model.parameters()):
                        net_param.data.copy_(new_param.cuda())
    
                    total_loss = total_loss * 0
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                else:
                    last_net = [x.detach().cpu() for x in model.parameters()]
                    psrn_noisy_last = noisy_psnr
    
            optimizer.step()
            optimizer.zero_grad()
    
            with torch.no_grad():
                out_full = model(noisy_in).detach().cpu()
                if H is not None:
                    out_full = out_full[:, :, :H, :W]
            if out_avg is None:
                out_avg = out_full.detach().cpu()
            else:
                out_avg = out_avg * exp_weight + out_full * (1 - exp_weight)
                out_avg = out_avg.detach().cpu()
    
        return out_avg


    if __name__ == "__main__":
        
        
        args, unknown = parser.parse_known_args()
        
        cfg = {'dev': False, 'experiment_cfg': {'LAM': 2, 'cuda': True, 'dataset': {'dataset_path': './data/PINCAT10', 'extension': 'png', 'gtandraw': True, 'resize': False}, 'input_type': 'noise', 'l1': 1e-05, 'lr': 0.0001, 'num_iter': 5500, 'optimizer': 'Adam', 'poisson_loss': True}, 'experiment_pipeline': 'ours', 'model_cfg': {'channels': 1, 'kernel_size': 3, 'norm': False, 'num_filters': 512, 'num_iter': 10, 'stride': 1, 'threshold': 0.01}, 'output_dir': './results/PINCAT10/'}
        
        noisy = imread(args.input_path)
        minner = np.amin(noisy)
        noisy = noisy-minner
        maxer = np.amax(noisy)
        noisy = noisy/maxer
        noisy = np.expand_dims(noisy,0)
        noisy = np.expand_dims(noisy,0)
        
        noisy = torch.from_numpy(noisy).float()
        
        
        
        out_image = main(noisy, cfg, cfg['experiment_cfg']) * maxer + minner
    
        imwrite(args.output_path,out_image.cpu().detach().numpy()) 
