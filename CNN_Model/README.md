This folder contains the scripts to train the model and classify images stored in `Final_Examples` folder.

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
   

**Final_Examples:** contains the examples which have been retained from the dataset. We can check if the model classifies them correctly or not using classify.py script

**pyimagesearch:** contains the smallervggnet model which we have used for image classification

**Final_Dataset_plot_0.8.png:** contains the plot which shows how training error, validation error, training accuracy and validation accuracy change in each epoch

**classify.py:** is the file which we will use to classify the examples located in Final_Examples directory. Use the following command to execute the script:
    
    $ python classify.py --model final_dataset_80.model --labelbin final_dataset_80_labels.pickle --image Final_Examples/<filename>.jpg

**final_dataset_80.model:** is the model we have trained on our system and can be used for classifying images as described above

**final_dataset_80_labels.pickle:** stores the labels of training data in pickle format which is used by the classify.py file to determine the class of image

**train.py:** is the script which can be used to train the model for your dataset. The dataset you want to train on should have the directory structure as below:
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
    
Refer the dataset we used for this model [here](https://indiana-my.sharepoint.com/personal/mloukil_iu_edu/_layouts/15/onedrive.aspx?slrid=e2f3a59e-706e-7000-5e41-de8ee166731f&id=%2fpersonal%2fmloukil_iu_edu%2fDocuments%2fDataset&FolderCTID=0x012000EBE26B4678EE41448F1B496277D1E812)

Use the following command to execute the script:
    
    $ python train.py --dataset <dataset_directory> --model <model_file_name>.model --labelbin <label_file_name>.pickle

    
    
References:
1. Keras and CNN tutorial by Adrian Rosebrock which can be referred [here](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/)