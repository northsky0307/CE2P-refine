# CE2P-refine# CE2P

### Requirements

To install PyTorch, please refer to https://github.com/pytorch/pytorch#installation.

### Compiling

Some parts of InPlace-ABN have a native CUDA implementation, which must be compiled with the following commands:
```bash
cd modules
sh build.sh
python build.py
``` 
The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

### Dataset and pretrained model

Plesae download cityscapes dataset and modify the `CS_PATH` in job_loacl.sh

Please download imagenet pretrained [resent-101](https://pan.baidu.com/s/1YMiL0lFgpzhIfD_IjwSJjw), and put it into dataset folder.

### Training 
```bash
./job_local.sh
``` 
