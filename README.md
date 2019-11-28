# [SFA: Small Faces Attention Face Detector](https://ieeexplore.ieee.org/document/8911451)

## Introduction
This repository includes the training and evaluating codes for *SFA* face detector, implemented in pycaffe. The SFA codes is an extension of the [SSH](https://github.com/mahyarnajibi/SSH) repository. Our method is trained on the training set of the WIDER FACE dataset.

## Contents
1. [Installation](#install)
2. [Configuration](#configuration)
3. [Preparation](#preparation)
4. [Demo](#demo)
5. [Training](#training)
6. [Evalution](#evaluation)

<a name="install"> </a>
### 1. Installation
1.1. Clone the repository:
```
git clone --recursive https://github.com/shiluo1990/SFA.git
```

1.2. Install [cuDNN](https://developer.nvidia.com/cudnn) and [NCCL](https://github.com/NVIDIA/nccl) (used for multi-GPU training).

1.3. Caffe and pycaffe: You need to compile the ```caffe-sfa``` repository which is a  Caffe fork compatible with *SFA*. Caffe should be built with *cuDNN*, *NCCL*, and *python layer support* (set by default in ```Makefile.config.example```). You also need to ```make pycaffe```.

1.4. Install python requirements:
```
pip install -r requirements.txt
```

1.5. Run ```make``` in the ```lib``` directory:
```
cd lib
make
```

<a name="configuration"> </a>
### 2. Configuration
#### 2.1. Default Configuration
All *SFA* default settings and configurations are saved in ```SFA/configs/default_config.yml```. More details can be seen in ```SFA/configs/default_config.yml``` as an example config files.

#### 2.2. Multi-scale Training and Testing Configuration
We provide 5 configuration files in ```SFA/configs/``` to perform MS-Training and MS-Testing in single_scale, 4_scale, and wide_scale. 
```
            single_scale                       4_scale                       wide_scale
training    wider_single_scale_training.yml    wider_4_scale_training.yml
testing     wider_single_scale_testing.yml     wider_4_scale_testing.yml     wider_wide_scale_testing.yml
```
They can be overwritten by passing an external configuration file to the module ```--cfg [path-to-config-file]```.
Note that you can adjust testing scales to flexibly trade off detection accuracy and efficiency for real applications. To avoid out of GPU memory, you can properly decrease the max scale in MS-Testing.
#### 2.3. Solver Configuration
Solver parameters are saved in ```SFA/models/solver_sfa.prototxt``` (*e.g.*  learning rate, gamma, stepsize, momentum, etc.). 

#### 2.4. Network Architecture Configuration
SFA training and testing network are described in
```
SFA/models/train_sfa.prototxt
SFA/models/test_sfa.prototxt
```

<a name="preparation"> </a>
### 3. Preparation
#### 3.1. Pre-trained SFA Model Download
To run the demo, you need to download the provided pre-trained *SFA* model from Baidu Yun via the following link and extraction code :
```
link: https://pan.baidu.com/s/1LAWhtiATbPHKweMeY8hVaQ
extraction code: ica9
```
By default, the model is saved into a folder named ```data/SFA_models```.

Before starting to run demo, you should have a directory structure as follows:
 ```
data
   |--demo
         |--demo.jpg
   |--SFA_models
         |--SFA.caffemodel
```

#### 3.2. Pre-trained VGG-16 ImageNet Download
To train your own SFA model, you also need to download the *VGG-16*  ImageNet model as pre-trained model. The following script downloads the model into the default directory:
```
bash scripts/download_imgnet_model.sh
```
If you can't download the pre-trained *VGG-16* ImageNet model via the above script, please try to download from Baidu Yun via the following link and extraction code :
```
link: https://pan.baidu.com/s/1VqiwWHiFPnDefMymgdgbdA
extraction code: 7790
```
By default, the model is saved into a folder named ```data/imagenet_models```.
#### 3.3. WIDER FACE Dataset Download
For training and evaluation on the *WIDER* dataset, you need to download the WIDER FACE dataset from the [WIDER FACE dataset website](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). These files should be copied into ```data/datasets/wider/```.

Before starting to train and evaluate, you should have a directory structure as follows:
 ```
data
   |--datasets
         |--wider
             |--WIDER_train/
             |--WIDER_val/
             |--WIDER_test/
             |--wider_face_split/
   |--imagenet_models
         |--VGG16.caffemodel
```

<a name="demo"></a>
### 4. Demo
After downloading the *SFA* model, you can run the demo with the default configuration as follows:
```
python demo.py
```
If everything goes well, the following detections should be saved as ```data/demo/demo_detections_SFA.png```.
<p align="center">
<img src="https://github.com/shiluo1990/SFA/blob/master/data/demo/github.png" width=400 >
</p>

For a list of possible options run: ```python demo.py --help```

More detection results in our paper are offered in ```paper_result/detection_result```.

<a name="training"></a>
### 5. Training
For training with the default parameters, you can call the ```main_train``` module with a list of GPU ids. As an example:
```
python main_train.py --gpus 0,1
```
For a list of all possible options run ```python main_train.py --help```.

By default, the models are saved into the ```output/[EXP_DIR]/[db_name]/``` folder (```EXP_DIR``` is set to ```sfa``` by default and can be changed through the configuration files,
and ```db_name``` would be ```wider_train``` in this case).

<a name="evaluation"></a>
### 6. Evaluation
The evaluation on the *WIDER* dataset is based on the official *WIDER* evaluation tool which requires *MATLAB*.
We performed evaluation with the default configuration by calling the ```main_test``` module:
```
python main_test.py --model [path-to-the-trained-model]
```
For a list of possible options run ```python main_test.py --help```. 

The evaluation outputs are saved into ```output/[EXP_DIR]/[db_name]/[net_name]``` (```EXP_DIR``` is set to ```sfa``` by default and can be changed by passing a config file, ```net_name``` can be directly passed to the module and is set to ```SFA``` by default, and ```db_name```  would be ```wider_val``` in this case). This includes the detections saved as text files in a folder named ```detections```,detections saved as a ```pickle``` file, and the ```WIDER``` evaluation plots saved in a folder named ```wider_plots```. 

Please note that the detections will be cached by default and will not be re-computed again (the caching can be disabled by passing the ```--no_cache``` argument.)

The evaluation results in our paper are offered in ```paper_result/evaluation```.

## Citation
If this project helps your research, please consider citing the following papers:
```
@inproceedings{shiluo2019sfa,
   title={{SFA}: Small Faces Attention Face Detector},
   author={Shi, Luo and Xiongfei, Li and Rui, Zhu and Xiaoli, Zhang},
   journal={Signal Processing: Image Communication},
   year={2019}
}

@inproceedings{najibi2017ssh,
  title={{SSH}: Single Stage Headless Face Detector},
  author={Najibi, Mahyar and Samangouei, Pouya and Chellappa, Rama and Davis, Larry},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}
```

## Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/shiluo1990/SFA/issues).

