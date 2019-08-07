# SoundNet_Pytorch
Soundnet model in Pytorch

![from soundnet](https://camo.githubusercontent.com/0b88af5c13ba987a17dcf90cd58816cf8ef04554/687474703a2f2f70726f6a656374732e637361696c2e6d69742e6564752f736f756e646e65742f736f756e646e65742e6a7067)

# Introduction
The code is for converting the pretrained [tensorflow soundnet model](https://github.com/eborboihuc/SoundNet-tensorflow) to pytorch model. So no training code for SoundNet model. The pretrained pytorch soundnet model is *sound8.pth*

# Prerequisites
1. Tensorflow (Only if .pth doesn't exist)
2. python 3.6 with numpy
3. pytorch 0.4+


# How to use
1. If the file *sound8.pth* has not been generated yet, follow the original instructions : [model](https://github.com/smallflyingpig/SoundNet_Pytorch.git) 

2. If audio preprocessing is required (ex : the sample rate is not 22.050 Hz),[utils.py](../master/utils.py) has a method for converting the indicated folder.

    > To convert a file:  `sox input.wav -r 22050 -c 1 ouput.wav` 
3. To extract a features vector use:
```python
audio,sr = load_audio(filepath)
    features = ex.extract_pytorch_feature(audio,'./soundnet/sound8.pth')   
    print([x.shape for x in features])
    
    ##extract vector
    conv = ex.extract_vector(features,idlayer) #features vector
```
Highlevel features: 
- conv5, idlayer = 4
- conv7, idlayer = 6

#### The temporal resolution
In order to find the  the temporal resolution `1/m` for each layer, the slope and the interception are calculated, which describes the relationship between the time in seconds and the number of channels of the `extract_feature_vector` method.
# Acknowledgments 
Mode for soundnet tensorflow model is ported from [soundnet_tensorflow](https://github.com/eborboihuc/SoundNet-tensorflow). Thanks for his works!


# reference
1. Yusuf Aytar, Carl Vondrick, and Antonio Torralba. "Soundnet: Learning sound representations from unlabeled video." Advances in Neural Information Processing Systems. 2016.
