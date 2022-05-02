# Face Video Deblurring via Iterative Prior Estimation and Adaptive Kernel Recovery

Code repo for the paper "Face Video Deblurring via Iterative Prior Estimation and Adaptive Kernel Recovery".

![](/images/main.png)
 
 ## Effects of the iterative refinement
Effects of iterative refinement is demonstrated in the below figure. Red and blue arrows show landmark
estimation and deblurring steps, respectively. We observe that estimated
landmarks (red dots, i.e. centers of estimated heatmaps) become closer to the
ground truth ones (green dots) as iteration increases, consequently leading to
better restoration.

![](/images/ab.png)

## Pretrained Models

You could download the pretrained model on 300VW dataset [here](https://drive.google.com/file/d/1Bf-Munz0J932hjz547grGGloX_Ix8H6H/view?usp=sharing). 

## Prerequisites

- Linux (tested on Ubuntu 14.04/16.04)
- CUDA 8.0/9.0/10.0
- gcc 4.9+
- Python 2.7+
- PyTorch 1.0+
- easydict
- tensorboardX
Note that if your CUDA is 10.2+, you need to modify KernelConv2D_cuda.cpp:
```
1. Uncomment: #include <c10/cuda/CUDAStream.h>
2. Modify: at::cuda::getCurrentCUDAStream() -> c10::cuda::getCurrentCUDAStream()
```

#### Installation

```
pip install -r requirements.txt
bash install.sh
```

## Usage

Download 300VW dataset followed the instructions in the official [website](https://ibug.doc.ic.ac.uk/resources/300-VW/), and modify the data index tree as following:

```
├── [DATASET_ROOT]
│   ├── train
│   │   ├── input
│   │   ├── GT
│   ├── test
│   │   ├── input
│   │   ├── GT
```

Then create the interpolated version as specified in the paper following the usage in [sepconv-slomo](https://github.com/sniklaus/sepconv-slomo). Train-test split follows from [Ren et al.](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ren_Face_Video_Deblurring_Using_3D_Facial_Priors_ICCV_2019_paper.pdf).

Use the following command to train the neural network:

```
python runner.py 
        --phase 'train'\
        --data [dataset path]\
        --out [output path]
```

Use the following command to test the neural network:

```
python runner.py \
        --phase 'test'\
        --weights './ckpt/best-ckpt.pth.tar'\
        --data [dataset path]\
        --out [output path]
```
Use the following command to resume training the neural network:

```
python runner.py 
        --phase 'resume'\
        --weights './ckpt/best-ckpt.pth.tar'\
        --data [dataset path]\
        --out [output path]
```
You can also use the following simple command, with changing the settings in config.py:

```
python runner.py
```

## Acknowledgement
The implementation is modified from STFAN, please consider citing them below if you find the code useful.

```
@inproceedings{zhou2019stfan,
  title={Spatio-Temporal Filter Adaptive Network for Video Deblurring},
  author={Zhou, Shangchen and Zhang, Jiawei and Pan, Jinshan and Xie, Haozhe and  Zuo, Wangmeng and Ren, Jimmy},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```

## Contact

For questions related to the code, please email to wangrong089@gmail.com