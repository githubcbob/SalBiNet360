## SalBiNet360: Saliency Prediction on 360° Images with Local-Global Bifurcated Deep Network 

### This is a PyTorch implementation.

## Prerequisites

- [Pytorch 0.4.1+](http://pytorch.org/)
- [torchvision](http://pytorch.org/)
- python 3.6.1
- matlab 9.2.0+
- PIL 4.1.1+
- numpy
- tqdm
- scipy

## Usage

### 1. Enter the repository

```shell
cd SalBiNet/
```

### 2. Download the dataset

Download the following datasets and unzip them into `data/global` folder.

* 

### 3. Download the pre-trained model

Download the following [pre-trained models](https://) into `pretrained` folder.

### 4. Preprocess the dataset

1. Set the `infolder` path in `preprocessing.m` correctly.

2. Run the matlab script for generating local patches.
```shell
matlab -nodesktop -nosplash -r preprocessing
```
3. After preprocessing the result data will be stored under `data/local` folder.

### 5. Test

For predicting global saliency maps:
```shell
python test.py --type global
```
For predicting local saliency maps:
```shell
python test.py --type local
```
to get the fused saliency maps:
```shell
matlab -nodesktop -nosplash -logfile -r post_processing
```


