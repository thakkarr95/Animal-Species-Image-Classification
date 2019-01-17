This folder contains the script used to train a baseline model and baseline model built for our dataset

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
   

**bottleneck_fc_model.h5:** contains weights for the model trained

**bottleneck_features_train.npy:** contains the training dataset in a format acceptable by the baseline model

**bottleneck_features_validation.npy:** contains the vlaidation dataset in a format acceptable by the baseline model

**CNN_Model_VGG16.py:** is the sript which is used to train the baseline model
   
The dataset which we used contains images from animals with attributes dataset available [here](http://cvml.ist.ac.at/AwA2/) 

Make sure you have the dataset splitted into folders 'Train_0.8' and 'Validation_0.8' before you run the script

Use the following command to execute the script:
    
    $ python CNN_Model_VGG16.py 

We also trained the model we have in `CNN_Model` folder with the above dataset but the results are equvalent to using 70% of the final dataset which has been illustrated in `Experiments_to_identify_optimal_dataset_size` folder.
  
References:
1. Keras tutorial on Building powerful image classification models using very little data which can be referred [here](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)