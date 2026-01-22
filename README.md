# RefineOT

Highly multiplexed imaging techniques are vital tools in biomedical research, used to study complex tissue structures and cellular interactions in both normal and disease states. Despite their capabilities, these techniques often suffer from significant noise and bleed-through. RefineOT is a plug-and-play, blind, zero-shot denoising and bleed-through correction toolkit. It learns directly from the input image with no training data, no spillover matrices, and no sensitive hyperparameters. It has shown strong performance on IMC and t-CyCIF datasets.

![RefineOT](https://github.com/jason-lequyer/RefineOT/blob/main/gitfig.png)

## Contents

- Fiji Plugin
  - Quick Start
  - How to Run in Fiji
  - Outputs
  - Command-Line Use
- Denoiser
  - Multi-channel data
  - 2D data
  - Reproducibility
- Troubleshooting
- FAQ

---

# Fiji Plugin

## Quick Start

1) Install Fiji and anaconda if needed
   - Download Fiji from https://fiji.sc/ and install it.
   - If you don't already have anaconda, install it by following instructions at this link: https://docs.anaconda.com/anaconda/install/.

2) Create a conda environment named rfot
   - Run the following in a terminal or Anaconda Prompt:
     ```bash
     conda create -n rfot -c conda-forge python=3.11 numpy scipy tifffile imagecodecs
     conda activate rfot
     python -c "import numpy,scipy,tifffile; print('RFOT env OK')"
     ```

3) Install the plugin into Fiji
   - Download this repository by clicking Code -> Download ZIP in the top right corner and unzip it on your computer
   - Copy `plugins/Debleed` into your Fiji plugins folder, for example you should end up with:
     - macOS: `Fiji.app/plugins/Debleed/`
     - Windows: `Fiji.app\\plugins\\Debleed\\`
     - Linux: `Fiji.app/plugins/Debleed/`
   - Restart Fiji or use Help -> Refresh Menus.

## How to Run in Fiji

1) Open the image in Fiji as a TIFF stack where each slice corresponds to a single channel. To do this for Opal and some other proprietary formats, you may need to use Plugins → Bio-Formats → Bio-Formats Importer, tick Split channels on the popup menu and select the image series that contains the true channel-by-channel data.
2) Run the plugin from the menu if visible:
   - Plugins -> Debleed -> Debleed_Run
   - If it does not appear, open `Debleed_Run.py` in Fiji Script Editor and click Run, or use Help -> Refresh Menus.
3) First it will ask you to select which channel(s) you want to debleed
4) Then it will ask you to group co-expressing channels together:
   - This menu allows you to group co-expressing channels so they do not debleed each other. Use the matrix to exclude channels that share real signal, not bleed-through.

   - The plugin proposes an initial auto-grouping using channel names and a built-in lookup table of known co-expressing channels in IMC. Review and adjust as needed.

5) Finally, it will ask you to enter some parameters:
   - Patch size, must be an even integer greater than or equal to 4. Smaller patch size leads to more aggressive and faster debleeding. Default is 16, best results at 24.
   - Conda env path: the plugin tries to prefill the path to the rfot environment. If it is blank, paste the full path to your env. Examples are in the FAQ below.
   - Keep the brightest heuristic: assumes brightest isntance of a signal is the true signal and calls anything dimmer bleed-through, if this assumption is invalid (which is commonly the case with Opal), turn off to have our brightness-invariant extrema based signal detection performed.
   - Ignore overexposed pixels: any pixel hitting the pixel intensity limit of the image format (e.g. 4096 for 12-bit) will be set to zero before debleeding. Rarely needed.

Click OK to start. A progress window shows elapsed time while each channel is processed.


## Outputs

- Single channel mode produces `..._Channel_<N>_debleed.tif`.
- All channels mode produces `..._debleed.tif`.
- The datatype matches your input. Outputs are ImageJ compatible TIFFs.


## Command-Line Use (Terminal / Cluster)

This repository includes a **terminal-friendly** `debleed.py` wrapper that allows Debleed to be run outside Fiji on a workstation or compute cluster (e.g. SLURM). The script processes one or more channels from a TIFF stack and writes one output TIFF per processed channel for optimal paralellization.

### Basic usage

From the repository “master directory” (the directory containing `debleed.py`):

```bash
cd <masterdirectoryname>
conda activate rfot
python debleed.py <path/to/stack.tif> <channel>
```

Example (debleed channel 21 of the included IMC test data):

```bash
cd <masterdirectoryname>
conda activate rfot
python debleed.py IMC_smallcrop/IMC_smallcrop.tif 21
```

Channel indexing is **1-based** (so `21` means the 21st channel in the TIFF stack).

### Multiple channels / all channels

You can process multiple channels using comma-separated values and ranges:

```bash
python debleed.py <path/to/stack.tif> 1,3-5
```

To process **all channels** in the stack:

```bash
python debleed.py <path/to/stack.tif> all
```
### Optional arguments

#### Co-expression / Veto Matrix (Hightly recommended to use)

You can provide a **veto (co-expression) matrix** specifying which channels should *not* be considered when debleeding a given target channel.

- See `IMC_smallcrop_withcsv/IMC_smallcrop.csv` for an example.
- The matrix must have the **same base name as the TIFF stack**, but with a `.csv` extension.

Example:

- TIFF stack: `IMC_smallcrop/IMC_smallcrop.tif`
- Veto matrix: `IMC_smallcrop/IMC_smallcrop.csv`

CSV format:
- Rows and columns correspond to channels by their order in the TIFF stack
- A `0` at position `(row = y, column = x)` means **channel x will NOT be used** when debleeding channel y
- A `1` means it **will** be used

Note:
- Header names are not strictly required; the program assumes row/column order corresponds to TIFF channel order.
- If names are included, ensure they match the TIFF channel ordering.
#### Patch size

Patch size can also be set as follows:

```bash
python debleed.py IMC_smallcrop/IMC_smallcrop.tif 21 --patsize 12
# (also accepts: -p 12)
```

#### Ignore overexposed pixels

To set saturated pixels to zero before debleeding:

```bash
python debleed.py <path/to/stack.tif> 21 --ignore_overexposed
```


#### Opal / Vectra Data

For Opal/Vectra multiplex IHC data, the keep-the-brightest heuristic can be suboptimal so we reccommend using the signal-based flag:

```bash
python debleed.py <path/to/stack.tif> 21 --signal_based
```

#### Outputs

For each processed channel `N`, the output is written next to the input TIFF stack and typically named:

```
<stack_basename>_Channel_<N>_debleed.tif
```

Example:

```
IMC_smallcrop/IMC_smallcrop_Channel_21_debleed.tif
```

### Debugging tips

To stream the underlying runner output and print the exact subprocess commands:

```bash
python debleed.py <path/to/stack.tif> 21 --verbose
```

If your runner scripts are stored in a non-standard location, you can specify it explicitly:

```bash
python debleed.py <path/to/stack.tif> 21 --runner_dir /path/to/Debleed/scripts
```

# Denoiser

## Multi-channel data
The denoiser is not part of the Fiji plugin, but you can run it from the command line. We first need to install pytorch to our env though. Follow the instructions here to get a command you can enter to install pytorch: https://pytorch.org/get-started/locally/. If you have a GPU, select one of the compute platforms that starts with 'CUDA'. The command the website spits out should start with 'pip3'. Enter that command into the terminal and press enter.

The denoiser can be run in the exact same way as the terminal version of the debleeder:

```python
cd <masterdirectoryname>
conda activate RFOT
python denoise.py IMC_smallcrop/IMC_smallcrop.tif 21
```

This should take under 30 minutes to run on a GPU with >32GB of memory.

## 2D Data

To denoise 2D images, create a folder in the master directory (the directory that contains debleed.py) and put your noisy images into it. Then open anaconda prompt/terminal and run the following:

```python
cd <masterdirectoryname>
conda activate RFOT
python denoise2D.py <noisyfolder>/<noisyimagename>
```
Replacing "masterdirectoryname" with the full path to the directory that contains denoise2D.py, replacing "noisyfolder" with the name of the folder containing images you want denoised and replacing "noisyimagename" with the name of the image file you want denoised. Results will be saved to the directory '<noisyolder>_denoised'. Issues may arise if using an image format that is not supported by the tifffile python package, to fix these issues you can open your images in ImageJ and re-save them as .tif (even if they were already .tif, this will convert them to ImageJ .tif).

## Reproducibility

To run anything beyond this point in the readme, we need to install another conda library:

```python
conda install anaconda::scikit-image=0.23.2
```

### Using RefineOT on provided datasets

To run RefineOT denoise on one of the noisy microscope images, open a terminal in the master directory and run:

```python
cd <masterdirectoryname>
python denoise2D.py Microscope_gaussianpoisson/1.tif
```
The denoised results will be in the directory 'Microscope_gaussianpoisson_denoised'.

To run RefineOT denoise on our other datasets we first need to add synthetic gasussian noise. For example to test RefineOT denoise on Set12 with sigma=25 gaussian noise, we would first: 
```python
cd <masterdirectoryname>
python add_gaussian_noise.py Set12 25
```
This will create the folder 'Set12_gaussian25' which we can now denoise:

```python
python denoise2D.py Set12_gaussian25/01.tif
```
Which returns the denoised results in a folder named 'Set12_gaussian25_denoised'.
  


### Calculate accuracy of RefineOT Denoise

To find the PSNR and SSIM between a folder containing denoised results and the corresponding folder containing known ground truths (e.g. Set12_gaussian25_denoised and Set12 if you followed above), we need to install one more conda package:

```python
conda activate rfot
conda install -c anaconda scikit-image=0.19.2
```

Now we measure accuracy with the code:
```terminal
cd <masterdirectoryname>
python compute_psnr_ssim.py Set12_gaussian25_denoised Set12 255
```

You can replace 'Set12' and 'Set12_gaussian25' with any pair of denoised/ground truth folders (order doesn't matter). Average PSNR and SSIM will be returned for the entire set.

The '255' at the end denotes the dynamic range of the image, in the case of the 8-bit images from Set12, '255' is a sensible value. For the Microscope data, '700' is a more sensible value and will replicate the results from our paper.
  

  
### Running compared methods

We can run DIP, Noise2Self, P2S and N2F+DOM in the RFOT environment:

```python
conda activate rfot
python DIP.py Microscope_gaussianpoisson
python N2S.py Microscope_gaussianpoisson
python P2S.py Microscope_gaussianpoisson
python N2FDOM.py Microscope_gaussianpoisson
```


---

# Troubleshooting

- The plugin cannot find rfot automatically on macOS
  - Run `conda env list` to see the path, for example `/opt/anaconda3/envs/rfot`. Paste that full path into the dialog.
  - The plugin also tries `~/miniforge3/envs/rfot`, `~/mambaforge/envs/rfot`, and other common locations. If none are found, manual entry works.

- Windows MKL error such as mkl_intel_thread.2.dll not found
  - Recommended: switch NumPy or SciPy to OpenBLAS builds:
    ```bat
    conda activate rfot
    conda install -y -c conda-forge "blas=*=openblas" numpy scipy
    ```
  - Keep MKL: install the MKL runtime and ensure DLL search works:
    ```bat
    conda activate rfot
    conda install -y -c defaults mkl intel-openmp mkl-service
    set CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1
    ```
  - The plugin adds `<env>\\Library\\bin` to PATH for the child process on Windows.


---

# FAQ

- Where is my env installed
  - Run `conda env list`. Use the full path shown for rfot.
  - Examples
    - macOS: `/opt/anaconda3/envs/rfot` or `~/miniforge3/envs/rfot`
    - Windows: `C:\\Users\\<you>\\miniconda3\\envs\\rfot` or `C:\\Users\\<you>\\anaconda3\\envs\\rfot`
    - Linux: `~/mambaforge/envs/rfot` or `~/miniconda3/envs/rfot`

- Can I use Mambaforge or Miniforge
  - Yes. The plugin only needs the env root path and the env must contain python, numpy, scipy, and tifffile.

- Which packages are required for the Fiji debleeder
  - python=3.11, numpy, scipy, tifffile
  - imagecodecs is optional but recommended for wider TIFF codec support

- Does the Fiji plugin use a GPU
  - No. The debleeder is CPU-based. However the separate denoiser scripts will require a GPU.

---
