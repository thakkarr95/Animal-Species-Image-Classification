# AML_Project_Species_Image_Classification
This repository is used to store all the code that we implemented for project done as a part of CSCI-P556 course at Indiana University Bloomington

The project uses Keras and Tensorflow for Image Classification based in Animal Species found in a given image

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

   
**Adversarial_Examples:** contains the scripts and adversrial examples generated for VGG16 model and it demonstrates that our CNN model based on smallervggnet still classifies the image correctly

**Base_Line_Model:** contains the scripts used to split our dataset into training and validaion sets. 2 baseline models were implemented. First based on VGG16 architecture and second using smallervggnet. The dataset used is available [here](http://cvml.ist.ac.at/AwA2/)

**CNN_Model:** contains the scripts to train a model and a script classify images using the model trained. We trained the model using 80% of the dataset available. This decision was based on the experiments performed in the `Experiments_to_identify_optimal_dataset_size` folder. The dataset used for this model can be obtained by running Split_Dataset.ipynb notebook from `Experiments_to_identify_optimal_dataset_size` folder with split = 0.8

**CNN_Visualisation**: contains the scripts used to understand how CNN work using visulaisation. It also contains a visualisation tool box by Jason Yosinski. **Note:** The visualisation toolbox requires python 2.7 while all the other scripts are for python 3.5

**Experiments_to_identify_optimal_dataset_size:** contains the scripts used to run experiments to identify the optimal datasetsize so that the model trains in less time without comprmising much on accuracy. The folder also contains the results obtained for all the experiments. We found that on using 80% of the given dataset, the model takes 2.73 hours to train and with validation. accuracy of 74% as compared to a model which was trined using the complete dataset which took 3.69 hours to run with validation accuracy of 76%
  
**Note:** The experiments were run on a machine with Intel i7-8750H CPU, 16 GB RAM and NVIDIA GTX 1060 GPU with 6 GB

**Webscraping**: Contains the script used to scrape the web for images and creating our dataset. 

The dataset which we used contains images scraped from the web, from animals with attributes datases available [here](http://cvml.ist.ac.at/AwA2/) and selected images from iNaturalist competition on Kaggle which can be found [here](https://www.kaggle.com/c/inaturalist-2018/data)

References:
1. Keras and CNN tutorial by Adrian Rosebrock which can be referred [here](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/)
2. Keras tutorial on Building powerful image classification models using very little data which can be referred [here](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
3. Machine Learning Adversrial git repository by Grégory Châtel which can be found [here](https://github.com/rodgzilla/machine_learning_adversarial_examples)
4. Deep Visualisation Toolbox by Jason Yosinski which can be found [here](https://github.com/yosinski/deep-visualization-toolbox)
5. Deep dream example by Keras team which can be found [here](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py)
6. Deep dream example by SnowMasaya which can be found [here](https://github.com/PacktPublishing/Deep-Learning-with-Keras/blob/master/Chapter07/deep-dream.py)
