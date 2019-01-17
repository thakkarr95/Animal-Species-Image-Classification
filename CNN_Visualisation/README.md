This folder contains the scripts and a toolbox used to understand how CNN work by visulaization

We need to install the following dependencies for Keras and Tensorflow to work smoothly:

1. Check if you have CUDA enabled GPU. You can check it from the list [here](https://developer.nvidia.com/cuda-gpus)
2. Install Microsoft Visual Studio. You can install it from [here](https://go.microsoft.com/fwlink/?LinkId=532606&clcid=0x409)
3. Get the CUDA toolkit. This project uses CUDA 8.0 which can be downloaded from [here](https://developer.nvidia.com/cuda-80-ga2-download-archive)
4. Set up cudaNN from [here](https://developer.nvidia.com/rdp/cudnn-download). You will need to create an account on NVIDIA's website in order to download the library
5. Install tensorflow 1.4.0 pip

   $ pip install tensorflow-gpu==1.4.0
   
6. Install keras 2.0.8 using pip

   $ pip install keras==2.0.8
   
7. Make sure everything works by typing the following commands in python:
   
   $ import tensorflow as tf
   
   $ sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
   

**data:** contains the images used as input in CNN_Visualise___deep_dream.py script.

**deep-visualization-toolbox:** contains all the files to run the visualization-toolbox. It has been cloned from a repository by Jason Yosinski which can be found [here](https://github.com/yosinski/deep-visualization-toolbox)

Please refer the Readme.md file in deep-visualization-toolbox to setup an environment for the toolbox. **Note**: It requires python 2.7 while the rest of our project uses python 3.5

**results:** stores the result images obtained from CNN_Visualise__deep_dream.py script


**CNN_Visualise__deep_dream.py:** is a script used to visualise different layers of inception_v3 model. The code used is from https://github.com/keras-team/keras/blob/master/examples/deep_dream.py

Use the following command to execute the script:

    $ python CNN_Visualise__deep_dream.py data/mypic.jpg results

**CNN_Visualise_dream.py:** is a script used to understand what VGG16 identifies the most in a given image from `data` folder.

Make sure you change the path to input image in the script before you run it. The code used is from https://github.com/PacktPublishing/Deep-Learning-with-Keras/blob/master/Chapter07/deep-dream.py

Use the following command to execute the script: (Changing the image to visualize should be done from the script CNN_Visualise_dream.py)

    $ python CNN_Visualise_dream.py

    
    
References:
1. Deep Visualisation Toolbox by Jason Yosinski which can be found [here](https://github.com/yosinski/deep-visualization-toolbox)
2. Deep dream example by Keras team which can be found [here](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py)
3. Deep dream example by SnowMasaya which can be found [here](https://github.com/PacktPublishing/Deep-Learning-with-Keras/blob/master/Chapter07/deep-dream.py)
