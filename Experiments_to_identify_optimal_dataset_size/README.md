This folder contains the script and models that were used in order to identify the optimal dataset size that has to be used for CNN_Model

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
   

**pyimagesearch:** contains the smallervggnet model which we have used for image classification

**final_dataset_10.model:** stores the model obtained when 10% of the original dataset was used

**final_dataset_10_labels.pickle:** stores the labels of when 10% of original dataset was used

**final_dataset_20.model:** stores the model obtained when 20% of the original dataset was used

**final_dataset_20_labels.pickle:** stores the labels of when 20% of original dataset was used

**final_dataset_30.model:** stores the model obtained when 30% of the original dataset was used

**final_dataset_30_labels.pickle:** stores the labels of when 30% of original dataset was used

**final_dataset_40.model:** stores the model obtained when 40% of the original dataset was used

**final_dataset_40_labels.pickle:** stores the labels of when 40% of original dataset was used

**final_dataset_50.model:** stores the model obtained when 50% of the original dataset was used

**final_dataset_50_labels.pickle:** stores the labels of when 50% of original dataset was used

**final_dataset_60.model:** stores the model obtained when 60% of the original dataset was used

**final_dataset_60_labels.pickle:** stores the labels of when 60% of original dataset was used

**final_dataset_70.model:** stores the model obtained when 70% of the original dataset was used

**final_dataset_70_labels.pickle:** stores the labels of when 70% of original dataset was used

**final_dataset_80.model:** stores the model obtained when 80% of the original dataset was used

**final_dataset_80_labels.pickle:** stores the labels of when 80% of original dataset was used

**final_dataset_90.model:** stores the model obtained when 90% of the original dataset was used

**final_dataset_90_labels.pickle:** stores the labels of when 90% of original dataset was used

**final_dataset_100.model:** stores the model obtained when 100% of the original dataset was used

**final_dataset_100_labels.pickle:** stores the labels of when 100% of original dataset was used

**Split_Dataset.ipynb:** is used to split the original dataset into 10%, 20%, .... 90%. The dataset is splitted such that x% of images from each animal class are taken so that the proportion of classes in each splitted sataset are same

**Model_Time_Accuracy.xlsx:** contains the results obtained for the number of hours taken to train the model and validation accuracy achieved for the given model

**train.py:** is the script which can be used to train the model for your dataset.The script splits the dataset in 80% train and 20% validation datasets. The dataset you want to train on should have the directory structure as below:
    ├── dataset
    ├── class_1
    ├── class_2
    ├── class_3
    ├── class_4 
    └── class_5
            .
            .
            .
    └── class_n
    
The dataset which we used contains images scraped from the web, from animals with attributes datases available [here](http://cvml.ist.ac.at/AwA2/) and selected images from iNaturalist competition on Kaggle which can be found [here](https://www.kaggle.com/c/inaturalist-2018/data) 

Refer the dataset we used for the experiments [here](https://indiana-my.sharepoint.com/personal/mloukil_iu_edu/_layouts/15/onedrive.aspx?slrid=e2f3a59e-706e-7000-5e41-de8ee166731f&id=%2fpersonal%2fmloukil_iu_edu%2fDocuments%2fDataset&FolderCTID=0x012000EBE26B4678EE41448F1B496277D1E812)

Use the following command to execute the script:
    
    $ python train.py --dataset <dataset_directory> --model <model_file_name>.model --labelbin <label_file_name>.pickle
    

**Note:** The experiments were run on a machine with Intel i7-8750H CPU, 16 GB RAM and NVIDIA GTX 1060 GPU 6 GB
    
    
References:
1. Keras and CNN tutorial by Adrian Rosebrock which can be referred [here](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/)