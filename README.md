# SoundNet_Pytorch
[Soundnet (Aytar et al. 2016)](http://soundnet.csail.mit.edu/) - UE Deep Learning IMT Atlantique

![from soundnet](https://camo.githubusercontent.com/0b88af5c13ba987a17dcf90cd58816cf8ef04554/687474703a2f2f70726f6a656374732e637361696c2e6d69742e6564752f736f756e646e65742f736f756e646e65742e6a7067)

# Introduction
This repository serves as a support for the course project on the Soundnet paper, in the context of the "Deep Learning UE" at IMT Atlantique. 

The repository includes: 
- the definition of the pytorch model
- the weights of the model
- a few useful functions to perform an inference on sounds and to extract features from the internal layers (see examples below)

# Prerequisites
1. python 3.6 with numpy
2. pytorch 0.4+
3. Torchaudio (part of pytorch installation)
4. Install other missing packages with `pip install` if they are missing


# How to use
1. Load an audio file. File can be of any format supported by torchaudio, but note that you may to resample to 22050 Hz which is the frequency at which SoundNet was trained. To load audio, you can directly use [`torchaudio.load()`](https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#loading-audio-data) and you can then resample with [`torchaudio.transforms.Resample`](https://pytorch.org/audio/stable/transforms.html#resample). See [torchaudio documentation](https://pytorch.org/audio/stable/index.html) for details on supported file formats and other technicalities related to audio. 

```python
    from torchaudio import load
    from torchaudio.transforms import Resample
    X,sr = load(audiofile)

    # Check that the sample rate is the same as the trained SoundNet
    if sr != 22050:
        transform = Resample(sr,22050)
        X = transform(X)

    # X is now a torch FloatTensor of shape (Number of Channels, Number of samples)
    # A stereophonic recording will have 2 channels ; SoundNet only accepts monophonic so we average the two channels if necessary
    if X.shape[0]>1:
        X = torch.mean(X,axis=0)
    
```

2. Perform inference
   
SoundNet predicts objects from the [Imagenet LSVRC 2012 Classification Challenge](https://www.image-net.org/challenges/LSVRC/2012/index.php) (1000 categories) and the [Places365 dataset](http://places2.csail.mit.edu/) (401 places).

```python
    from pytorch_model import SoundNet8_pytorch
    from utils import vector_to_scenes,vector_to_obj

    ## define the soundnet model
    model = SoundNet8_pytorch()
    ## Load the weights of the pretrained model
    model.load_state_dict(torch.load('sound8.pth'))
    # Reshape the data to the format expected by the model (Batch, 1, time, 1)
    X = X.view(1,1,-1,1)
    
    # Compute the predictions of Objects and Scenes
    object_pred, scene_pred = model(X)

    ## Shape of object_pred is (1,1000,TIME,1) as there are 1000 possible objects
    ## Find the correspond object labels, and print it for each time point

    print(vector_to_obj(object_pred.detach().numpy()))

    # Shape of scene_pred is (1,401,TIME,1) as there are 401 possible scenes
    ## Find the correspond scene label, and print it  for each time point

    print(vector_to_scenes(scene_pred.detach().numpy()))
```

3. Extract features vectors from the internal SoundNet layers
```python    
    # Extract internal features
    features = model.extract_feat(X)

    # Features is a List of Tensors, each element of this list corresponds to a layer of SoundNet. From 0 to 6 -> conv1 to conv7, 7 -> conv of object prediction and 8 -> conv of scene prediction. See the extract_feat method in the model code.

    ## Example : Feature maps of Layer Conv5 is of shape : (Batch, Units, Time, 1)
    print((features[4].shape))
```

Note that the model works on Batches of sounds of the same duration ; in all examples above we have used a batch of size 1 but you can load sets of sounds for batch processing.

# Possible tasks for the project 
1. Supervised learning on a small dataset (e.g. [ESC10 / ESC50](https://github.com/karolpiczak/ESC-50) or another small dataset such as [GTZAN Music Genre Classification](https://pytorch.org/audio/stable/datasets.html#gtzan) by extracting feature vectors from the internal layers.
2. Study the sensitivity of various units to different sound types.
3. Any other result from the paper that you find interesting
4. Fine-tuning on a small set of videos : for that you have to define and load [pretrained ImageNet models](https://pytorch.org/vision/0.8/models.html#classification) and [Places365 models](https://github.com/CSAILVision/places365), and extract probabilities (see SoundNet paper).

Don't hesitate to ask to confirm the steps needed to perform those tasks. 

# Acknowledgments 
Mode for soundnet tensorflow model is ported from [soundnet_tensorflow](https://github.com/eborboihuc/SoundNet-tensorflow). Thanks for his works!


# reference
1. Yusuf Aytar, Carl Vondrick, and Antonio Torralba. "Soundnet: Learning sound representations from unlabeled video." Advances in Neural Information Processing Systems. 2016.
