DparNet: Degradation parameter assisted restoration network
=============
## Description
This is the implementation of manuscript "Wide & deep learning for 
spatial & intensity adaptive image restoration". 

## System requirements
#### Prerequisites
* Ubuntu 18.04+ or Windows 10/11
* NVIDIA GPU + CUDA (Geforce RTX 3090-Ti with 24GB memory, 
CUDA 11.1 was tested)

#### Installation
* Python 3.7+
* Pytorch 1.7.0+

## Quick start
#### Dataset
We provided very small version of the constructed 
denoising and deturbulence dataset, deposited in 
```./denoising/data/``` and ```./deturbulence/data/```.

#### Test and evaluation
* Run ```python test.py``` in each sub-folder to run the 
pre-trained models. The restored results will be saved in 
```./results/``` of each sub-folder.
* Run ```python evaluate.py``` in each sub-folder to obtain the 
quantitative evaluations of the restored results. 
As very little data is provided here, 
the evaluation results will not be the same as in manuscript.

#### Training
* Run ```python train.py``` to perform training with the 
default setting.