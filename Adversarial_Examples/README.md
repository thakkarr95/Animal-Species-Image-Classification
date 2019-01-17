This folder contains the Adverserial Examples that have been generated

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
   

**adversarial_images:** contains the adverserial images generated using Adverserial_Image_Generation.ipynb notebook. The code used is from https://github.com/rodgzilla/machine_learning_adversarial_examples

**Final_Examples:** contains the images for which adverserial examples will be generated

**pyimagesearch:** contains the smallervggnet model which we have used for image classification

**Adverserial_Image_Generation.ipynb:** is the notebook we will use to generate adverserial images for VGG16 model

**classify.py:** is a script used to classify the adverserial examples using our smallervggnet model

Steps to execute the scripts:
1. Use Adverserial_Image_Generation.ipynb to generate adversrial examples for VGG16 model. Make sure that the adverserial images are misclassified by VGG16 model in the notebook
2. Use the generated adversarial examples in classify.py file to see if the images are still classified correctly by our smallervggnet model

Run the script using the command below:

    $ python classify.py --model final_dataset_80.model --labelbin final_dataset_80_labels.pickle --image adversarial_images/<filename>.jpg
   
References:
1. Keras and CNN tutorial by Adrian Rosebrock which can be referred [here](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/)
2. Machine Learning Adversrial git repository by Grégory Châtel which can be found [here](https://github.com/rodgzilla/machine_learning_adversarial_examples)
