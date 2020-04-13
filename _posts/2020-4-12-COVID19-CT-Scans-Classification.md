<h1 align=center><font size = 5>COVID19 CT Scans Classifications</font></h1>

## Introduction


In this project, we are going to investigate the performance of some Deep Learning Neural Networks for classification of Computed Tomography (CT) scans.

## Business Understanding


We seek to investigate whether Deep Learning models can effectively speed up the identification of COVID19 patients by looking at CT scans of Lungs.  For this to be a fruitful endeavor, the models have to perform well in order to provide confidence in the classification results.  For this purpose, we will look at the following measures:

  * **Accuracy** measures how often the classifier makes the correct prediction. It is  the ratio of the number of correct predictions to the total number of predictions (the number of test data points).

  * **Precision** tells us what proportion of CT scans we classified as positive, actually were positive.  It is a ratio of true positives(scans classified as positive, and which are actually positive) to all positives(all scans classified as positive, irrespective of whether that was the correct classification).

  * **Recall(sensitivity)** tells us what proportion of scans that actually were positive were classified by us as positive.  It is a ratio of true positives(scans classified as positive, and which are actually positive) to all the scans that were actually positive.

  * **F-beta score** is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.  The $\beta$ parameter determines the weight of recall in the combined score. beta < 1 lends more weight to precision, while beta > 1 favors recall (beta -> 0 considers only precision, beta -> +inf only recall).  In particular, when $\beta = 0.5$, more emphasis is placed on precision. This is called the **F$_{0.5}$ score** (or F-score for simplicity).

## Table of Contents

<div class="alert alert-block alert-info" style="margin-top: 20px">

<font size = 3>    

1. <a href="#item41">Data Understanding 
2. <a href="#item42">Data Preparation and Modeling</a>
3. <a href="#item43">Evaluation</a>  
4. <a href="#item44">Conclusions</a>  

</font>
    
</div>


```python
# Colab recommend against using pip install to specify a particular TensorFlow version for both GPU and TPU backends. 
# Colab builds TensorFlow from source to ensure compatibility with our fleet of accelerators. 
#Versions of TensorFlow fetched from PyPI by pip may suffer from performance problems or may not work at all.
#!pip install tensorflow-gpu==1.15

%tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)
```

    TensorFlow 1.x selected.
    1.15.2



```python
#Numpy contains helpful functions for efficient mathematical calculations
import numpy as np
#Dataframe manipulation library
import pandas as pd

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

#Graph plotting library
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

from PIL import Image

import os, random

#import covid19_datasets_utils as utils

import zipfile
from zipfile import ZipFile

import keras
from keras.models import Sequential
from keras.layers import Dense

#import the ImageDataGenerator module since we will be leveraging it to train our model in batches.
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

from keras.models import load_model

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

from keras.applications import DenseNet121
from keras.applications.densenet import preprocess_input as densenet121_preprocess_input

from keras.applications import NASNetMobile
from keras.applications.nasnet import preprocess_input as nasnetmobile_preprocess_input

from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix

import covid19_datasets_utils as utils

```

<a id="item41"></a>

## Data Understanding

We are using the datasets from [COVID-CT](https://github.com/UCSD-AI4H/COVID-CT) to evaluate the performances of the various models.  

In [here](https://github.com/UCSD-AI4H/COVID-CT/tree/master/Images-processed), there are two zip files - CT_COVID.zip and CT_NonCOVID.zip.  CT_COVID.zip is a zip file containing all the postive samples of CT scans of COVID19 patient, whereas, CT_NonCOVID.zip the negative samples.  The directory structure of Data_split is organized as follows:   

```
  +-- Data-split
  |   +-- COVID
  |   |   +-- testCT_COVID.txt
  |   |   +-- trainCT_COVID.txt
  |   |   +-- valCT_COVID.txt
  |   +-- NonCOVID
  |   |   +-- testCT_NonCOVID.txt
  |   |   +-- trainCT_NonCOVID.txt
  |   |   +-- valCT_NonCOVID.txt
  ```

  Each of the above text files contain the filename of the samples.


```python
# running utils.main() will set up the datasets correctly
# Note that the current directory should have 3 zip files -- CT_COVID.zip, CT_NonCOVID.zip and Data-split.zip
# before running the "main" function
utils.main()
```

    All the files in the current directory: ['.config', 'CT_NonCOVID.zip', 'covid19_datasets_utils.py', 'Data-split.zip', '__pycache__', 'data_processed', 'CT_COVID.zip', 'data', 'sample_data']
    All files zipped successfully!


After you unzip the data, you will find the data has already been divided into a train, validation, and test sets.  The directory structure will be as follows:   

```
  +-- data_processed
  |   +-- test
  |   |   +-- COVID
  |   |   +-- NonCOVID
  |   +-- train
  |   |   +-- COVID
  |   |   +-- NonCOVID
  |   +-- valid
  |   |   +-- COVID
  |   |   +-- NonCOVID
```   

The training dataset has 191 COVID (positive) samples and 234 NonCOVID (negative) samples.  The test dataset 95 positive and 105 negative samples.  The validation dataset 58 positive and 58 negative samples.  Using pretrained models would be appropriate since these are small datasets.


```python
# unzip the data_processed.zip file
with zipfile.ZipFile("data_processed.zip", "r") as zip_ref:
  zip_ref.extractall('./')

```


```python
# Show original picture
img_dir = './data_processed/train/COVID/'
 
# randomly pick an image
img = random.choice(os.listdir(img_dir))
img_path = img_dir + img
 
with Image.open(img_path) as image1:
    plt.imshow(image1)
    print(image1.mode)
```

    RGBA



![png]({{ site.baseurl }}/images/covid19_udacity/output_14_1.png)


<a id="item42"></a>

## Data Preparation and Modeling

We are going to use five pre-trained models (VGG16, ResNet50, DenseNet121, NASNetMobile and InceptionV3) from [keras](https://keras.io/applications/), and determine which model will give the best performance using the above measures.

In subsequent sections, we are only going to mention about VGG16 pre-trained model.  The rest of the four models are following exactly the same procedure.  We can import the model <code>VGG16</code> from <code>keras.applications</code>.

We essentially build your classifier as follows:
1. Import libraries, modules, and packages you will need. Make sure to import the *preprocess_input* function from <code>keras.applications.vgg16</code>.
2. Use a batch size of 100 images for both training and validation.
3. Construct an ImageDataGenerator for the training set and another one for the validation set. VGG16 was originally trained on 224 × 224 images, so make sure to address that when defining the ImageDataGenerator instances.  All the five pre-trained models use 224 × 224 images, except InceptionV3 where 299 x 299 images are used.
4. Create a sequential model using Keras. Add VGG16 model to it and dense layer.
5. Compile the mode using the adam optimizer and the categorical_crossentropy loss function.
6. Fit the model on the augmented data using the ImageDataGenerators.   

Note that all the images are scaled to match the input sizes of the models.  The training datasets are also agumented with shearing, zooming and horizontal_flip to make the final models more robust.

#### Define Global Constants
Here, we will define constants that we will be using throughout the rest of the lab. 

1. We are obviously dealing with two classes, so *num_classes* is 2. 
2. We will training and validating the model using batches of 100 images.
3. The total number of epochs for training each model be 100.


```python
num_classes = 2

batch_size_training = 100
batch_size_validation = 100

num_epochs = 100
```

#### Define our Models

In this section, we will define our five models.


```python
# VGG16 model
def create_vgg16_model(num_classes=num_classes, image_resize=224, batch_size_training=batch_size_training, batch_size_validation=batch_size_validation ):
  '''
  INPUT:
  num_classes - number of clases (should be 2 here)
  image_resize - image size of the pre-trained model
  batch_size_training - batch training size
  batch_size_validation - batch validation size


  OUTPUT:
  model - Our custom model from the pre-trained model
  train_generator - Train data generator
  validation_generator - validation data generator
    
  This function create a custom model using the pre-trained model, by removing the last layer of the pre-trained model
  and adding the "Dense" layer with 2 outputs.
  '''
  # Construct ImageDataGenerator Instances
  # In order to instantiate an ImageDataGenerator instance, we will set the **preprocessing_function** argument 
  # to *preprocess_input* which we imported from **keras.applications.vgg16** in order to preprocess our images 
  #the same way the images used to train VGG16 model were processed.

  train_data_generator = ImageDataGenerator(
    preprocessing_function=vgg16_preprocess_input,
    #rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )
  
  data_generator = ImageDataGenerator(
    preprocessing_function=vgg16_preprocess_input
    )
 
  # Next, we will use the flow_from_directory method to get the training images as follows:
  train_generator = train_data_generator.flow_from_directory(
    './data_processed/train',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical',
    shuffle=True,
    seed=42
    )
  
  # Use the flow_from_directory method to get the validation images and assign the result to validation_generator.
  validation_generator = data_generator.flow_from_directory(
    './data_processed/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical')
  
  # Build, Compile and Fit Model   
  # In this section, we will start building our model. We will use the Sequential model class from Keras.
  model = Sequential()

  # Next, we will add the VGG16 pre-trained model to out model. However, note that we don't want to include 
  #the top layer or the output layer of the pre-trained model. We actually want to define our own output layer 
  #and train it so that it is optimized for our image dataset. In order to leave out the output layer of the pre-trained model, 
  #we will use the argument include_top and set it to False.

  model.add(VGG16(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))
  
  # Then, we will define our output layer as a Dense layer, that consists of two nodes 
  #and uses the Softmax function as the activation function.

  model.add(Dense(num_classes, activation='softmax'))

  # Since the VGG16 model has already been trained, then we want to tell our model not to bother with training the VGG16 part, 
  #but to train only our dense output layer. 
  model.layers[0].trainable = False

  # And now using the summary attribute of the model, we can see how many parameters 
  # we will need to optimize in order to train the output layer.
  model.summary()

  # compile our model using the adam optimizer.
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model, train_generator, validation_generator

```


```python
# ResNet50 model
def create_resnet50_model(num_classes=num_classes, image_resize=224, batch_size_training=batch_size_training, batch_size_validation=batch_size_validation):
  '''
  INPUT:
  num_classes - number of clases (should be 2 here)
  image_resize - image size of the pre-trained model
  batch_size_training - batch training size
  batch_size_validation - batch validation size


  OUTPUT:
  model - Our custom model from the pre-trained model
  train_generator - Train data generator
  validation_generator - validation data generator
    
  This function create a custom model using the pre-trained model, by removing the last layer of the pre-trained model
  and adding the "Dense" layer with 2 outputs.
  '''
  # Construct ImageDataGenerator Instances
  # In order to instantiate an ImageDataGenerator instance, we will set the **preprocessing_function** argument 
  # to *preprocess_input* which we imported from **keras.applications.resnet** in order to preprocess our images 
  # the same way the images used to train ResNet50 model were processed.

  train_data_generator = ImageDataGenerator(
    preprocessing_function=resnet50_preprocess_input,
    #rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )
  
  data_generator = ImageDataGenerator(
    preprocessing_function=resnet50_preprocess_input
    )
 
  # Next, we will use the flow_from_directory method to get the training images as follows:
  train_generator = train_data_generator.flow_from_directory(
    './data_processed/train',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical',
    shuffle=True,
    seed=42
    )
  
  # Use the flow_from_directory method to get the validation images and assign the result to validation_generator.
  validation_generator = data_generator.flow_from_directory(
    './data_processed/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical')
  
  # Build, Compile and Fit Model   
  # In this section, we will start building our model. We will use the Sequential model class from Keras.
  model = Sequential()

  # Next, we will add the VGG16 pre-trained model to out model. However, note that we don't want to include 
  #the top layer or the output layer of the pre-trained model. We actually want to define our own output layer 
  #and train it so that it is optimized for our image dataset. In order to leave out the output layer of the pre-trained model, 
  #we will use the argument include_top and set it to False.

  model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))
 
  # Then, we will define our output layer as a Dense layer, that consists of two nodes 
  #and uses the Softmax function as the activation function.

  model.add(Dense(num_classes, activation='softmax'))

  # Since the VGG16 model has already been trained, then we want to tell our model not to bother with training the VGG16 part, 
  #but to train only our dense output layer. 
  model.layers[0].trainable = False

  # And now using the summary attribute of the model, we can see how many parameters 
  # we will need to optimize in order to train the output layer.
  model.summary()

  # compile our model using the adam optimizer.
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model, train_generator, validation_generator

```


```python
# DenseNet121 model
def create_densenet121_model(num_classes=num_classes, image_resize=224, batch_size_training=batch_size_training, batch_size_validation=batch_size_validation):
  '''
  INPUT:
  num_classes - number of clases (should be 2 here)
  image_resize - image size of the pre-trained model
  batch_size_training - batch training size
  batch_size_validation - batch validation size


  OUTPUT:
  model - Our custom model from the pre-trained model
  train_generator - Train data generator
  validation_generator - validation data generator
    
  This function create a custom model using the pre-trained model, by removing the last layer of the pre-trained model
  and adding the "Dense" layer with 2 outputs.
  '''
  # Construct ImageDataGenerator Instances
  # In order to instantiate an ImageDataGenerator instance, we will set the **preprocessing_function** argument 
  # to *preprocess_input* which we imported from **keras.applications.densenet** in order to preprocess our images 
  #the same way the images used to train DenseNet121 model were processed.

  train_data_generator = ImageDataGenerator(
    preprocessing_function=densenet121_preprocess_input,
    #rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )
  
  data_generator = ImageDataGenerator(
    preprocessing_function=densenet121_preprocess_input
    )
 
  # Next, we will use the flow_from_directory method to get the training images as follows:
  train_generator = train_data_generator.flow_from_directory(
    './data_processed/train',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical',
    shuffle=True,
    seed=42
    )
  
  # Use the flow_from_directory method to get the validation images and assign the result to validation_generator.
  validation_generator = data_generator.flow_from_directory(
    './data_processed/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical')
  
  # Build, Compile and Fit Model   
  # In this section, we will start building our model. We will use the Sequential model class from Keras.
  model = Sequential()

  # Next, we will add the VGG16 pre-trained model to out model. However, note that we don't want to include 
  #the top layer or the output layer of the pre-trained model. We actually want to define our own output layer 
  #and train it so that it is optimized for our image dataset. In order to leave out the output layer of the pre-trained model, 
  #we will use the argument include_top and set it to False.

  model.add(DenseNet121(
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg',
    weights='imagenet',
    ))
  
  # Then, we will define our output layer as a Dense layer, that consists of two nodes 
  #and uses the Softmax function as the activation function.

  model.add(Dense(num_classes, activation='softmax'))

  # Since the VGG16 model has already been trained, then we want to tell our model not to bother with training the VGG16 part, 
  #but to train only our dense output layer. 
  model.layers[0].trainable = False

  # And now using the summary attribute of the model, we can see how many parameters 
  # we will need to optimize in order to train the output layer.
  model.summary()

  # compile our model using the adam optimizer.
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model, train_generator, validation_generator

```


```python
# NASNetMobile model
def create_nasnetmobile_model(num_classes=num_classes, image_resize=224, batch_size_training=batch_size_training, batch_size_validation=batch_size_validation):
  '''
  INPUT:
  num_classes - number of clases (should be 2 here)
  image_resize - image size of the pre-trained model
  batch_size_training - batch training size
  batch_size_validation - batch validation size


  OUTPUT:
  model - Our custom model from the pre-trained model
  train_generator - Train data generator
  validation_generator - validation data generator
    
  This function create a custom model using the pre-trained model, by removing the last layer of the pre-trained model
  and adding the "Dense" layer with 2 outputs.
  '''
  # Construct ImageDataGenerator Instances
  # In order to instantiate an ImageDataGenerator instance, we will set the **preprocessing_function** argument 
  # to *preprocess_input* which we imported from **keras.applications.nasnet** in order to preprocess our images 
  #the same way the images used to train NASNetMobile  model were processed.

  train_data_generator = ImageDataGenerator(
    preprocessing_function=nasnetmobile_preprocess_input,
    #rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )
  
  data_generator = ImageDataGenerator(
    preprocessing_function=nasnetmobile_preprocess_input
    )
 
  # Next, we will use the flow_from_directory method to get the training images as follows:
  train_generator = train_data_generator.flow_from_directory(
    './data_processed/train',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical',
    shuffle=True,
    seed=42
    )
  
  # Use the flow_from_directory method to get the validation images and assign the result to validation_generator.
  validation_generator = data_generator.flow_from_directory(
    './data_processed/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical')
  
  # Build, Compile and Fit Model   
  # In this section, we will start building our model. We will use the Sequential model class from Keras.
  model = Sequential()

  # Next, we will add the VGG16 pre-trained model to out model. However, note that we don't want to include 
  #the top layer or the output layer of the pre-trained model. We actually want to define our own output layer 
  #and train it so that it is optimized for our image dataset. In order to leave out the output layer of the pre-trained model, 
  #we will use the argument include_top and set it to False.

  model.add(NASNetMobile(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))
  
  # Then, we will define our output layer as a Dense layer, that consists of two nodes 
  #and uses the Softmax function as the activation function.

  model.add(Dense(num_classes, activation='softmax'))

  # Since the VGG16 model has already been trained, then we want to tell our model not to bother with training the VGG16 part, 
  #but to train only our dense output layer. 
  model.layers[0].trainable = False

  # And now using the summary attribute of the model, we can see how many parameters 
  # we will need to optimize in order to train the output layer.
  model.summary()

  # compile our model using the adam optimizer.
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model, train_generator, validation_generator

```


```python
# InceptionV3 model
def create_inceptionv3_model(num_classes=num_classes, image_resize=299, batch_size_training=batch_size_training, batch_size_validation=batch_size_validation):
  '''
  INPUT:
  num_classes - number of clases (should be 2 here)
  image_resize - image size of the pre-trained model
  batch_size_training - batch training size
  batch_size_validation - batch validation size


  OUTPUT:
  model - Our custom model from the pre-trained model
  train_generator - Train data generator
  validation_generator - validation data generator
    
  This function create a custom model using the pre-trained model, by removing the last layer of the pre-trained model
  and adding the "Dense" layer with 2 outputs.
  '''
  # Construct ImageDataGenerator Instances
  # In order to instantiate an ImageDataGenerator instance, we will set the **preprocessing_function** argument 
  # to *preprocess_input* which we imported from **keras.applications.inception_v3** in order to preprocess our images 
  #the same way the images used to train InceptionV3 model were processed.

  train_data_generator = ImageDataGenerator(
    preprocessing_function=inceptionv3_preprocess_input,
    #rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )
  
  data_generator = ImageDataGenerator(
    preprocessing_function=inceptionv3_preprocess_input
    )
 
  # Next, we will use the flow_from_directory method to get the training images as follows:
  train_generator = train_data_generator.flow_from_directory(
    './data_processed/train',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical',
    shuffle=True,
    seed=42
    )
  
  # Use the flow_from_directory method to get the validation images and assign the result to validation_generator.
  validation_generator = data_generator.flow_from_directory(
    './data_processed/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical')
  
  # Build, Compile and Fit Model   
  # In this section, we will start building our model. We will use the Sequential model class from Keras.
  model = Sequential()

  # Next, we will add the VGG16 pre-trained model to out model. However, note that we don't want to include 
  #the top layer or the output layer of the pre-trained model. We actually want to define our own output layer 
  #and train it so that it is optimized for our image dataset. In order to leave out the output layer of the pre-trained model, 
  #we will use the argument include_top and set it to False.

  model.add(InceptionV3(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))
  
  # Then, we will define our output layer as a Dense layer, that consists of two nodes 
  #and uses the Softmax function as the activation function.

  model.add(Dense(num_classes, activation='softmax'))

  # Since the VGG16 model has already been trained, then we want to tell our model not to bother with training the VGG16 part, 
  #but to train only our dense output layer. 
  model.layers[0].trainable = False

  # And now using the summary attribute of the model, we can see how many parameters 
  # we will need to optimize in order to train the output layer.
  model.summary()

  # compile our model using the adam optimizer.
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model, train_generator, validation_generator

```

Next we compile our model using the adam optimizer.


```python
model_vgg16, vgg16_train_generator, vgg16_validation_generator = create_vgg16_model(num_classes=2, image_resize=224, batch_size_training=100, batch_size_validation=100)
```

    Found 425 images belonging to 2 classes.
    Found 116 images belonging to 2 classes.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.
    
    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58892288/58889256 [==============================] - 2s 0us/step
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    vgg16 (Model)                (None, 512)               14714688  
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 1026      
    =================================================================
    Total params: 14,715,714
    Trainable params: 1,026
    Non-trainable params: 14,714,688
    _________________________________________________________________



```python

model_resnet50, resnet50_train_generator, resnet50_validation_generator = create_resnet50_model(num_classes=2, image_resize=224, batch_size_training=100, batch_size_validation=100)
```

    Found 425 images belonging to 2 classes.
    Found 116 images belonging to 2 classes.
    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    94658560/94653016 [==============================] - 3s 0us/step
    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    resnet50 (Model)             (None, 2048)              23587712  
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 4098      
    =================================================================
    Total params: 23,591,810
    Trainable params: 4,098
    Non-trainable params: 23,587,712
    _________________________________________________________________



```python
model_densenet121, densenet121_train_generator, densenet121_validation_generator = create_densenet121_model(num_classes=2, image_resize=224, batch_size_training=100, batch_size_validation=100)
```

    Found 425 images belonging to 2 classes.
    Found 116 images belonging to 2 classes.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4074: The name tf.nn.avg_pool is deprecated. Please use tf.nn.avg_pool2d instead.
    
    Downloading data from https://github.com/keras-team/keras-applications/releases/download/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
    29089792/29084464 [==============================] - 1s 0us/step
    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    densenet121 (Model)          (None, 1024)              7037504   
    _________________________________________________________________
    dense_3 (Dense)              (None, 2)                 2050      
    =================================================================
    Total params: 7,039,554
    Trainable params: 2,050
    Non-trainable params: 7,037,504
    _________________________________________________________________



```python
model_nasnetmobile, nasnetmobile_train_generator, nasnetmobile_validation_generator = create_nasnetmobile_model(num_classes=2, image_resize=224, batch_size_training=100, batch_size_validation=100)
```

    Found 425 images belonging to 2 classes.
    Found 116 images belonging to 2 classes.
    Downloading data from https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/NASNet-mobile-no-top.h5
    19996672/19993432 [==============================] - 1s 0us/step
    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    NASNet (Model)               (None, 1056)              4269716   
    _________________________________________________________________
    dense_4 (Dense)              (None, 2)                 2114      
    =================================================================
    Total params: 4,271,830
    Trainable params: 2,114
    Non-trainable params: 4,269,716
    _________________________________________________________________



```python
model_inceptionv3, inceptionv3_train_generator, inceptionv3_validation_generator = create_inceptionv3_model(num_classes=2, image_resize=299, batch_size_training=100, batch_size_validation=100)
```

    Found 425 images belonging to 2 classes.
    Found 116 images belonging to 2 classes.
    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    87916544/87910968 [==============================] - 2s 0us/step
    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    inception_v3 (Model)         (None, 2048)              21802784  
    _________________________________________________________________
    dense_5 (Dense)              (None, 2)                 4098      
    =================================================================
    Total params: 21,806,882
    Trainable params: 4,098
    Non-trainable params: 21,802,784
    _________________________________________________________________



```python
def train_model(model, train_generator, validation_generator, model_filepath, num_epochs=num_epochs):
  '''
  INPUT:
  model - Our custom model from the pre-trained model
  train_generator - Train data generator
  validation_generator - validation data generator
  model_filepath - filename to store the best classifier (based on validation accuracy)
  num_epochs = maximum number of epochs for training the custom model

  OUTPUT:
  fit_history - dictionary containing statistics of the training of the model

  This function trains the custom model, saves the best classifier and return the training statistics.
  '''
  # Before we are able to start the training process, with an ImageDataGenerator, 
  # we will need to define how many steps compose an epoch. 
  # Typically, that is the number of images divided by the batch size.

  steps_per_epoch_training = len(train_generator)
  steps_per_epoch_validation = len(validation_generator)

  # Finally, we are ready to start training our model. 
  # Unlike a conventional deep learning training were data is not streamed from a directory, 
  # with an ImageDataGenerator where data is augmented in batches, we use the fit_generator method.

  # At each epoch, Keras checks if our model performed better than the models of the previous epochs. 
  # If it is the case, the new best model weights are saved into a file. 
  # This will allow us to load directly the weights of our model without having to re-train it 
  # if we want to use it in another situation
  checkpoint = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]

  fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
    callbacks=callbacks_list
  )

  # Since training can take a long time when building deep learning models, 
  # it is always a good idea to save your model once the training is complete 
  #if you believe you will be using the model again later. 
  #model.save(model_filepath)

  return fit_history
  
```


```python
def plot_training_loss_and_accuracy(fit_history, title, save_filepath):
  '''
  INPUT:
  fit_history - dictionary containing statistics of the training of the model
  title = title for the plot
  save_filepath - filepath to save the plot

  OUTPUT:
  None

  This function display 2 plots side-by-side - Training Loss vs Validation Loss
  and Training Accuracy vs Validation Accuracy, and store the plot as an png file.
  '''
  fig = plt.figure(figsize=(20,10))
  ax1 = plt.subplot(1, 2, 1)
  plt.suptitle(title, fontsize=10)
  plt.ylabel('Loss', fontsize=16)
  plt.plot(fit_history.history['loss'], label='Training Loss')
  plt.plot(fit_history.history['val_loss'], label='Validation Loss')
  plt.legend(loc='upper right')

  # Hide the right and top spines
  ax1.spines['right'].set_visible(False)
  ax1.spines['top'].set_visible(False)

  ax2 = plt.subplot(1, 2, 2)
  plt.ylabel('Accuracy', fontsize=16)
  plt.plot(fit_history.history['accuracy'], label='Training Accuracy')
  plt.plot(fit_history.history['val_accuracy'], label='Validation Accuracy')
  plt.legend(loc='lower right')

  # Hide the right and top spines
  ax2.spines['right'].set_visible(False)
  ax2.spines['top'].set_visible(False)

  plt.show()

  fig.savefig(save_filepath, dpi=200, format='png', bbox_inches='tight')
```

Finally, we are ready to start training our model.


```python
vgg16_fit_history = train_model(model_vgg16, vgg16_train_generator, vgg16_validation_generator, 'classifier_vgg16_model.h5', num_epochs=num_epochs)
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    
    Epoch 1/100
    5/5 [==============================] - 18s 4s/step - loss: 4.2485 - accuracy: 0.5224 - val_loss: 2.8277 - val_accuracy: 0.4138
    
    Epoch 00001: val_accuracy improved from -inf to 0.41379, saving model to classifier_vgg16_model.h5
    Epoch 2/100
    5/5 [==============================] - 3s 615ms/step - loss: 2.7246 - accuracy: 0.4753 - val_loss: 2.9618 - val_accuracy: 0.5086
    
    Epoch 00002: val_accuracy improved from 0.41379 to 0.50862, saving model to classifier_vgg16_model.h5
    Epoch 3/100
    5/5 [==============================] - 7s 1s/step - loss: 2.5948 - accuracy: 0.4612 - val_loss: 1.1944 - val_accuracy: 0.4569
    
    Epoch 00003: val_accuracy did not improve from 0.50862
    Epoch 4/100
    5/5 [==============================] - 7s 1s/step - loss: 2.0729 - accuracy: 0.5247 - val_loss: 3.8547 - val_accuracy: 0.5000
    
    Epoch 00004: val_accuracy did not improve from 0.50862
    Epoch 5/100
    5/5 [==============================] - 7s 1s/step - loss: 1.9265 - accuracy: 0.5882 - val_loss: 1.5883 - val_accuracy: 0.5345
    
    Epoch 00005: val_accuracy improved from 0.50862 to 0.53448, saving model to classifier_vgg16_model.h5
    Epoch 6/100
    5/5 [==============================] - 7s 1s/step - loss: 1.5808 - accuracy: 0.5765 - val_loss: 0.9116 - val_accuracy: 0.5000
    
    Epoch 00006: val_accuracy did not improve from 0.53448
    Epoch 7/100
    5/5 [==============================] - 7s 1s/step - loss: 1.4561 - accuracy: 0.5765 - val_loss: 2.6231 - val_accuracy: 0.5517
    
    Epoch 00007: val_accuracy improved from 0.53448 to 0.55172, saving model to classifier_vgg16_model.h5
    Epoch 8/100
    5/5 [==============================] - 7s 1s/step - loss: 1.1165 - accuracy: 0.6518 - val_loss: 2.1130 - val_accuracy: 0.5776
    
    Epoch 00008: val_accuracy improved from 0.55172 to 0.57759, saving model to classifier_vgg16_model.h5
    Epoch 9/100
    5/5 [==============================] - 7s 1s/step - loss: 1.2494 - accuracy: 0.6165 - val_loss: 1.8824 - val_accuracy: 0.5862
    
    Epoch 00009: val_accuracy improved from 0.57759 to 0.58621, saving model to classifier_vgg16_model.h5
    Epoch 10/100
    5/5 [==============================] - 7s 1s/step - loss: 1.0740 - accuracy: 0.6329 - val_loss: 1.4553 - val_accuracy: 0.5603
    
    Epoch 00010: val_accuracy did not improve from 0.58621
    Epoch 11/100
    5/5 [==============================] - 7s 1s/step - loss: 0.9739 - accuracy: 0.6776 - val_loss: 0.5215 - val_accuracy: 0.5776
    
    Epoch 00011: val_accuracy did not improve from 0.58621
    Epoch 12/100
    5/5 [==============================] - 7s 1s/step - loss: 0.9570 - accuracy: 0.6588 - val_loss: 1.0106 - val_accuracy: 0.5776
    
    Epoch 00012: val_accuracy did not improve from 0.58621
    Epoch 13/100
    5/5 [==============================] - 7s 1s/step - loss: 0.9778 - accuracy: 0.6965 - val_loss: 2.2459 - val_accuracy: 0.5862
    
    Epoch 00013: val_accuracy did not improve from 0.58621
    Epoch 14/100
    5/5 [==============================] - 7s 1s/step - loss: 0.7955 - accuracy: 0.7059 - val_loss: 1.3128 - val_accuracy: 0.5862
    
    Epoch 00014: val_accuracy did not improve from 0.58621
    Epoch 15/100
    5/5 [==============================] - 7s 1s/step - loss: 0.8192 - accuracy: 0.6941 - val_loss: 1.1541 - val_accuracy: 0.6034
    
    Epoch 00015: val_accuracy improved from 0.58621 to 0.60345, saving model to classifier_vgg16_model.h5
    Epoch 16/100
    5/5 [==============================] - 7s 1s/step - loss: 0.7817 - accuracy: 0.7153 - val_loss: 1.2405 - val_accuracy: 0.6207
    
    Epoch 00016: val_accuracy improved from 0.60345 to 0.62069, saving model to classifier_vgg16_model.h5
    Epoch 17/100
    5/5 [==============================] - 7s 1s/step - loss: 0.7411 - accuracy: 0.7318 - val_loss: 1.1285 - val_accuracy: 0.5862
    
    Epoch 00017: val_accuracy did not improve from 0.62069
    Epoch 18/100
    5/5 [==============================] - 7s 1s/step - loss: 0.6247 - accuracy: 0.7529 - val_loss: 0.7206 - val_accuracy: 0.5948
    
    Epoch 00018: val_accuracy did not improve from 0.62069
    Epoch 19/100
    5/5 [==============================] - 7s 1s/step - loss: 0.6592 - accuracy: 0.7388 - val_loss: 1.6845 - val_accuracy: 0.6034
    
    Epoch 00019: val_accuracy did not improve from 0.62069
    Epoch 20/100
    5/5 [==============================] - 7s 1s/step - loss: 0.6384 - accuracy: 0.7600 - val_loss: 0.8556 - val_accuracy: 0.5948
    
    Epoch 00020: val_accuracy did not improve from 0.62069
    Epoch 21/100
    5/5 [==============================] - 7s 1s/step - loss: 0.7161 - accuracy: 0.7318 - val_loss: 0.6001 - val_accuracy: 0.6034
    
    Epoch 00021: val_accuracy did not improve from 0.62069
    Epoch 22/100
    5/5 [==============================] - 7s 1s/step - loss: 0.6596 - accuracy: 0.7624 - val_loss: 0.7737 - val_accuracy: 0.6293
    
    Epoch 00022: val_accuracy improved from 0.62069 to 0.62931, saving model to classifier_vgg16_model.h5
    Epoch 23/100
    5/5 [==============================] - 7s 1s/step - loss: 0.6766 - accuracy: 0.7506 - val_loss: 2.0297 - val_accuracy: 0.6207
    
    Epoch 00023: val_accuracy did not improve from 0.62931
    Epoch 24/100
    5/5 [==============================] - 7s 1s/step - loss: 0.6072 - accuracy: 0.7506 - val_loss: 1.3905 - val_accuracy: 0.6034
    
    Epoch 00024: val_accuracy did not improve from 0.62931
    Epoch 25/100
    5/5 [==============================] - 7s 1s/step - loss: 0.6045 - accuracy: 0.7835 - val_loss: 0.7201 - val_accuracy: 0.6034
    
    Epoch 00025: val_accuracy did not improve from 0.62931
    Epoch 26/100
    5/5 [==============================] - 7s 1s/step - loss: 0.5147 - accuracy: 0.7812 - val_loss: 1.8825 - val_accuracy: 0.6034
    
    Epoch 00026: val_accuracy did not improve from 0.62931
    Epoch 27/100
    5/5 [==============================] - 7s 1s/step - loss: 0.6588 - accuracy: 0.7553 - val_loss: 0.6181 - val_accuracy: 0.5948
    
    Epoch 00027: val_accuracy did not improve from 0.62931
    Epoch 28/100
    5/5 [==============================] - 7s 1s/step - loss: 0.5665 - accuracy: 0.7765 - val_loss: 0.9029 - val_accuracy: 0.6379
    
    Epoch 00028: val_accuracy improved from 0.62931 to 0.63793, saving model to classifier_vgg16_model.h5
    Epoch 29/100
    5/5 [==============================] - 7s 1s/step - loss: 0.5235 - accuracy: 0.7882 - val_loss: 0.9511 - val_accuracy: 0.6121
    
    Epoch 00029: val_accuracy did not improve from 0.63793
    Epoch 30/100
    5/5 [==============================] - 7s 1s/step - loss: 0.5148 - accuracy: 0.7694 - val_loss: 1.1798 - val_accuracy: 0.6293
    
    Epoch 00030: val_accuracy did not improve from 0.63793
    Epoch 31/100
    5/5 [==============================] - 7s 1s/step - loss: 0.5421 - accuracy: 0.7694 - val_loss: 0.8321 - val_accuracy: 0.6207
    
    Epoch 00031: val_accuracy did not improve from 0.63793
    Epoch 32/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4125 - accuracy: 0.8188 - val_loss: 0.8128 - val_accuracy: 0.6293
    
    Epoch 00032: val_accuracy did not improve from 0.63793
    Epoch 33/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4511 - accuracy: 0.8071 - val_loss: 1.1105 - val_accuracy: 0.6379
    
    Epoch 00033: val_accuracy did not improve from 0.63793
    Epoch 34/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4762 - accuracy: 0.7882 - val_loss: 0.6129 - val_accuracy: 0.6121
    
    Epoch 00034: val_accuracy did not improve from 0.63793
    Epoch 35/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4760 - accuracy: 0.7976 - val_loss: 1.1760 - val_accuracy: 0.6379
    
    Epoch 00035: val_accuracy did not improve from 0.63793
    Epoch 36/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4907 - accuracy: 0.7976 - val_loss: 0.8123 - val_accuracy: 0.6121
    
    Epoch 00036: val_accuracy did not improve from 0.63793
    Epoch 37/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3830 - accuracy: 0.8165 - val_loss: 1.2478 - val_accuracy: 0.6552
    
    Epoch 00037: val_accuracy improved from 0.63793 to 0.65517, saving model to classifier_vgg16_model.h5
    Epoch 38/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4526 - accuracy: 0.8024 - val_loss: 0.7634 - val_accuracy: 0.6379
    
    Epoch 00038: val_accuracy did not improve from 0.65517
    Epoch 39/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4121 - accuracy: 0.8353 - val_loss: 1.0167 - val_accuracy: 0.6207
    
    Epoch 00039: val_accuracy did not improve from 0.65517
    Epoch 40/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4506 - accuracy: 0.8306 - val_loss: 0.5092 - val_accuracy: 0.6293
    
    Epoch 00040: val_accuracy did not improve from 0.65517
    Epoch 41/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4666 - accuracy: 0.8188 - val_loss: 0.7293 - val_accuracy: 0.6207
    
    Epoch 00041: val_accuracy did not improve from 0.65517
    Epoch 42/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4509 - accuracy: 0.8282 - val_loss: 0.9868 - val_accuracy: 0.6121
    
    Epoch 00042: val_accuracy did not improve from 0.65517
    Epoch 43/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4220 - accuracy: 0.8329 - val_loss: 1.3623 - val_accuracy: 0.6121
    
    Epoch 00043: val_accuracy did not improve from 0.65517
    Epoch 44/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4551 - accuracy: 0.8376 - val_loss: 1.1692 - val_accuracy: 0.6293
    
    Epoch 00044: val_accuracy did not improve from 0.65517
    Epoch 45/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4466 - accuracy: 0.8212 - val_loss: 0.9723 - val_accuracy: 0.6466
    
    Epoch 00045: val_accuracy did not improve from 0.65517
    Epoch 46/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4132 - accuracy: 0.8141 - val_loss: 1.1946 - val_accuracy: 0.6293
    
    Epoch 00046: val_accuracy did not improve from 0.65517
    Epoch 47/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3773 - accuracy: 0.8447 - val_loss: 0.4151 - val_accuracy: 0.6638
    
    Epoch 00047: val_accuracy improved from 0.65517 to 0.66379, saving model to classifier_vgg16_model.h5
    Epoch 48/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4046 - accuracy: 0.8353 - val_loss: 1.6073 - val_accuracy: 0.6552
    
    Epoch 00048: val_accuracy did not improve from 0.66379
    Epoch 49/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4156 - accuracy: 0.8353 - val_loss: 1.4760 - val_accuracy: 0.6552
    
    Epoch 00049: val_accuracy did not improve from 0.66379
    Epoch 50/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3387 - accuracy: 0.8541 - val_loss: 0.3542 - val_accuracy: 0.6638
    
    Epoch 00050: val_accuracy did not improve from 0.66379
    Epoch 51/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3812 - accuracy: 0.8541 - val_loss: 0.6597 - val_accuracy: 0.6466
    
    Epoch 00051: val_accuracy did not improve from 0.66379
    Epoch 52/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3410 - accuracy: 0.8588 - val_loss: 0.4164 - val_accuracy: 0.6552
    
    Epoch 00052: val_accuracy did not improve from 0.66379
    Epoch 53/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3845 - accuracy: 0.8471 - val_loss: 0.5564 - val_accuracy: 0.6638
    
    Epoch 00053: val_accuracy did not improve from 0.66379
    Epoch 54/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3806 - accuracy: 0.8329 - val_loss: 1.1942 - val_accuracy: 0.6810
    
    Epoch 00054: val_accuracy improved from 0.66379 to 0.68103, saving model to classifier_vgg16_model.h5
    Epoch 55/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3620 - accuracy: 0.8612 - val_loss: 1.4740 - val_accuracy: 0.6724
    
    Epoch 00055: val_accuracy did not improve from 0.68103
    Epoch 56/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3260 - accuracy: 0.8588 - val_loss: 0.7317 - val_accuracy: 0.6810
    
    Epoch 00056: val_accuracy did not improve from 0.68103
    Epoch 57/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4199 - accuracy: 0.8612 - val_loss: 0.4965 - val_accuracy: 0.6810
    
    Epoch 00057: val_accuracy did not improve from 0.68103
    Epoch 58/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3238 - accuracy: 0.8518 - val_loss: 0.8529 - val_accuracy: 0.6897
    
    Epoch 00058: val_accuracy improved from 0.68103 to 0.68966, saving model to classifier_vgg16_model.h5
    Epoch 59/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3707 - accuracy: 0.8565 - val_loss: 0.8598 - val_accuracy: 0.6810
    
    Epoch 00059: val_accuracy did not improve from 0.68966
    Epoch 60/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3453 - accuracy: 0.8471 - val_loss: 0.3908 - val_accuracy: 0.6897
    
    Epoch 00060: val_accuracy did not improve from 0.68966
    Epoch 61/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3375 - accuracy: 0.8706 - val_loss: 1.3141 - val_accuracy: 0.6810
    
    Epoch 00061: val_accuracy did not improve from 0.68966
    Epoch 62/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2802 - accuracy: 0.8682 - val_loss: 0.5469 - val_accuracy: 0.6638
    
    Epoch 00062: val_accuracy did not improve from 0.68966
    Epoch 63/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3030 - accuracy: 0.8424 - val_loss: 1.4433 - val_accuracy: 0.6897
    
    Epoch 00063: val_accuracy did not improve from 0.68966
    Epoch 64/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3392 - accuracy: 0.8706 - val_loss: 0.8513 - val_accuracy: 0.6897
    
    Epoch 00064: val_accuracy did not improve from 0.68966
    Epoch 65/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3053 - accuracy: 0.8729 - val_loss: 1.2084 - val_accuracy: 0.6810
    
    Epoch 00065: val_accuracy did not improve from 0.68966
    Epoch 66/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3242 - accuracy: 0.8659 - val_loss: 1.0065 - val_accuracy: 0.6897
    
    Epoch 00066: val_accuracy did not improve from 0.68966
    Epoch 67/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2970 - accuracy: 0.8847 - val_loss: 0.6348 - val_accuracy: 0.6724
    
    Epoch 00067: val_accuracy did not improve from 0.68966
    Epoch 68/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3088 - accuracy: 0.8682 - val_loss: 1.1179 - val_accuracy: 0.6638
    
    Epoch 00068: val_accuracy did not improve from 0.68966
    Epoch 69/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3069 - accuracy: 0.8800 - val_loss: 1.4410 - val_accuracy: 0.6897
    
    Epoch 00069: val_accuracy did not improve from 0.68966
    Epoch 70/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2642 - accuracy: 0.8847 - val_loss: 1.2728 - val_accuracy: 0.6897
    
    Epoch 00070: val_accuracy did not improve from 0.68966
    Epoch 71/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2519 - accuracy: 0.8918 - val_loss: 0.8521 - val_accuracy: 0.6897
    
    Epoch 00071: val_accuracy did not improve from 0.68966
    Epoch 72/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2869 - accuracy: 0.8824 - val_loss: 0.4325 - val_accuracy: 0.6638
    
    Epoch 00072: val_accuracy did not improve from 0.68966
    Epoch 73/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3370 - accuracy: 0.8612 - val_loss: 1.0208 - val_accuracy: 0.6897
    
    Epoch 00073: val_accuracy did not improve from 0.68966
    Epoch 74/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2631 - accuracy: 0.8588 - val_loss: 0.9357 - val_accuracy: 0.6897
    
    Epoch 00074: val_accuracy did not improve from 0.68966
    Epoch 75/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2642 - accuracy: 0.8941 - val_loss: 1.3675 - val_accuracy: 0.6897
    
    Epoch 00075: val_accuracy did not improve from 0.68966
    Epoch 76/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2644 - accuracy: 0.8894 - val_loss: 0.1988 - val_accuracy: 0.6810
    
    Epoch 00076: val_accuracy did not improve from 0.68966
    Epoch 77/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3175 - accuracy: 0.8635 - val_loss: 0.9662 - val_accuracy: 0.6810
    
    Epoch 00077: val_accuracy did not improve from 0.68966
    Epoch 78/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2805 - accuracy: 0.9012 - val_loss: 1.1243 - val_accuracy: 0.6983
    
    Epoch 00078: val_accuracy improved from 0.68966 to 0.69828, saving model to classifier_vgg16_model.h5
    Epoch 79/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2757 - accuracy: 0.8894 - val_loss: 1.3310 - val_accuracy: 0.6897
    
    Epoch 00079: val_accuracy did not improve from 0.69828
    Epoch 80/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2589 - accuracy: 0.8824 - val_loss: 0.6809 - val_accuracy: 0.6983
    
    Epoch 00080: val_accuracy did not improve from 0.69828
    Epoch 81/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2990 - accuracy: 0.8894 - val_loss: 0.4638 - val_accuracy: 0.6983
    
    Epoch 00081: val_accuracy did not improve from 0.69828
    Epoch 82/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2601 - accuracy: 0.8847 - val_loss: 0.9995 - val_accuracy: 0.6983
    
    Epoch 00082: val_accuracy did not improve from 0.69828
    Epoch 83/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2489 - accuracy: 0.8871 - val_loss: 1.0986 - val_accuracy: 0.6897
    
    Epoch 00083: val_accuracy did not improve from 0.69828
    Epoch 84/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2581 - accuracy: 0.8824 - val_loss: 1.7784 - val_accuracy: 0.6983
    
    Epoch 00084: val_accuracy did not improve from 0.69828
    Epoch 85/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2695 - accuracy: 0.9012 - val_loss: 0.4063 - val_accuracy: 0.6983
    
    Epoch 00085: val_accuracy did not improve from 0.69828
    Epoch 86/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2431 - accuracy: 0.9059 - val_loss: 0.9727 - val_accuracy: 0.6897
    
    Epoch 00086: val_accuracy did not improve from 0.69828
    Epoch 87/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2376 - accuracy: 0.8988 - val_loss: 1.0620 - val_accuracy: 0.6897
    
    Epoch 00087: val_accuracy did not improve from 0.69828
    Epoch 88/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2415 - accuracy: 0.8918 - val_loss: 0.9801 - val_accuracy: 0.7069
    
    Epoch 00088: val_accuracy improved from 0.69828 to 0.70690, saving model to classifier_vgg16_model.h5
    Epoch 89/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2743 - accuracy: 0.8847 - val_loss: 0.8512 - val_accuracy: 0.7155
    
    Epoch 00089: val_accuracy improved from 0.70690 to 0.71552, saving model to classifier_vgg16_model.h5
    Epoch 90/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2599 - accuracy: 0.8941 - val_loss: 0.9101 - val_accuracy: 0.6983
    
    Epoch 00090: val_accuracy did not improve from 0.71552
    Epoch 91/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2714 - accuracy: 0.8635 - val_loss: 1.0503 - val_accuracy: 0.7069
    
    Epoch 00091: val_accuracy did not improve from 0.71552
    Epoch 92/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2686 - accuracy: 0.8729 - val_loss: 0.6554 - val_accuracy: 0.6983
    
    Epoch 00092: val_accuracy did not improve from 0.71552
    Epoch 93/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2059 - accuracy: 0.9082 - val_loss: 0.4307 - val_accuracy: 0.7069
    
    Epoch 00093: val_accuracy did not improve from 0.71552
    Epoch 94/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2745 - accuracy: 0.8918 - val_loss: 1.1165 - val_accuracy: 0.7069
    
    Epoch 00094: val_accuracy did not improve from 0.71552
    Epoch 95/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2175 - accuracy: 0.8871 - val_loss: 0.5703 - val_accuracy: 0.6897
    
    Epoch 00095: val_accuracy did not improve from 0.71552
    Epoch 96/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2075 - accuracy: 0.9294 - val_loss: 0.7668 - val_accuracy: 0.7069
    
    Epoch 00096: val_accuracy did not improve from 0.71552
    Epoch 97/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2286 - accuracy: 0.9129 - val_loss: 1.3011 - val_accuracy: 0.7069
    
    Epoch 00097: val_accuracy did not improve from 0.71552
    Epoch 98/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2237 - accuracy: 0.8871 - val_loss: 1.2695 - val_accuracy: 0.6983
    
    Epoch 00098: val_accuracy did not improve from 0.71552
    Epoch 99/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2038 - accuracy: 0.9035 - val_loss: 0.7854 - val_accuracy: 0.6897
    
    Epoch 00099: val_accuracy did not improve from 0.71552
    Epoch 100/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2321 - accuracy: 0.9082 - val_loss: 0.6729 - val_accuracy: 0.6983
    
    Epoch 00100: val_accuracy did not improve from 0.71552



```python
plot_training_loss_and_accuracy(vgg16_fit_history, 'Evolution of loss and accuracy with the number of training epochs for VGG16', 'vgg16_loss_acc.png')
```


![png]({{ site.baseurl }}/images/covid19_udacity/output_37_0.png)



```python
resnet50_fit_history = train_model(model_resnet50, resnet50_train_generator, resnet50_validation_generator, 'classifier_resnet50_model.h5', num_epochs=num_epochs)
```

    Epoch 1/100
    5/5 [==============================] - 9s 2s/step - loss: 0.7450 - accuracy: 0.5365 - val_loss: 1.1053 - val_accuracy: 0.5086
    
    Epoch 00001: val_accuracy improved from -inf to 0.50862, saving model to classifier_resnet50_model.h5
    Epoch 2/100
    5/5 [==============================] - 3s 516ms/step - loss: 0.5794 - accuracy: 0.6635 - val_loss: 0.5998 - val_accuracy: 0.5431
    
    Epoch 00002: val_accuracy improved from 0.50862 to 0.54310, saving model to classifier_resnet50_model.h5
    Epoch 3/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4987 - accuracy: 0.7506 - val_loss: 0.7771 - val_accuracy: 0.5603
    
    Epoch 00003: val_accuracy improved from 0.54310 to 0.56034, saving model to classifier_resnet50_model.h5
    Epoch 4/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4544 - accuracy: 0.7694 - val_loss: 0.8492 - val_accuracy: 0.5690
    
    Epoch 00004: val_accuracy improved from 0.56034 to 0.56897, saving model to classifier_resnet50_model.h5
    Epoch 5/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4411 - accuracy: 0.7953 - val_loss: 0.7076 - val_accuracy: 0.5776
    
    Epoch 00005: val_accuracy improved from 0.56897 to 0.57759, saving model to classifier_resnet50_model.h5
    Epoch 6/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4121 - accuracy: 0.8165 - val_loss: 0.6347 - val_accuracy: 0.5776
    
    Epoch 00006: val_accuracy did not improve from 0.57759
    Epoch 7/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4059 - accuracy: 0.8353 - val_loss: 0.5553 - val_accuracy: 0.5948
    
    Epoch 00007: val_accuracy improved from 0.57759 to 0.59483, saving model to classifier_resnet50_model.h5
    Epoch 8/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3488 - accuracy: 0.8447 - val_loss: 0.6350 - val_accuracy: 0.5776
    
    Epoch 00008: val_accuracy did not improve from 0.59483
    Epoch 9/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3766 - accuracy: 0.8353 - val_loss: 0.9246 - val_accuracy: 0.6293
    
    Epoch 00009: val_accuracy improved from 0.59483 to 0.62931, saving model to classifier_resnet50_model.h5
    Epoch 10/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3276 - accuracy: 0.8706 - val_loss: 0.7135 - val_accuracy: 0.6121
    
    Epoch 00010: val_accuracy did not improve from 0.62931
    Epoch 11/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3163 - accuracy: 0.8635 - val_loss: 0.8133 - val_accuracy: 0.5862
    
    Epoch 00011: val_accuracy did not improve from 0.62931
    Epoch 12/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3175 - accuracy: 0.8824 - val_loss: 0.7744 - val_accuracy: 0.6379
    
    Epoch 00012: val_accuracy improved from 0.62931 to 0.63793, saving model to classifier_resnet50_model.h5
    Epoch 13/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2857 - accuracy: 0.8776 - val_loss: 0.8881 - val_accuracy: 0.6293
    
    Epoch 00013: val_accuracy did not improve from 0.63793
    Epoch 14/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2986 - accuracy: 0.8918 - val_loss: 0.7823 - val_accuracy: 0.5603
    
    Epoch 00014: val_accuracy did not improve from 0.63793
    Epoch 15/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2883 - accuracy: 0.8800 - val_loss: 0.6407 - val_accuracy: 0.6121
    
    Epoch 00015: val_accuracy did not improve from 0.63793
    Epoch 16/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2958 - accuracy: 0.8800 - val_loss: 0.5835 - val_accuracy: 0.6034
    
    Epoch 00016: val_accuracy did not improve from 0.63793
    Epoch 17/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2751 - accuracy: 0.8659 - val_loss: 0.8870 - val_accuracy: 0.5776
    
    Epoch 00017: val_accuracy did not improve from 0.63793
    Epoch 18/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2660 - accuracy: 0.9059 - val_loss: 0.9743 - val_accuracy: 0.6552
    
    Epoch 00018: val_accuracy improved from 0.63793 to 0.65517, saving model to classifier_resnet50_model.h5
    Epoch 19/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2848 - accuracy: 0.8847 - val_loss: 1.0976 - val_accuracy: 0.5948
    
    Epoch 00019: val_accuracy did not improve from 0.65517
    Epoch 20/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2374 - accuracy: 0.9106 - val_loss: 1.0395 - val_accuracy: 0.6121
    
    Epoch 00020: val_accuracy did not improve from 0.65517
    Epoch 21/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2407 - accuracy: 0.9106 - val_loss: 0.6256 - val_accuracy: 0.6207
    
    Epoch 00021: val_accuracy did not improve from 0.65517
    Epoch 22/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2537 - accuracy: 0.8965 - val_loss: 0.8782 - val_accuracy: 0.5259
    
    Epoch 00022: val_accuracy did not improve from 0.65517
    Epoch 23/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2521 - accuracy: 0.8988 - val_loss: 1.0605 - val_accuracy: 0.6293
    
    Epoch 00023: val_accuracy did not improve from 0.65517
    Epoch 24/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2109 - accuracy: 0.9247 - val_loss: 0.8581 - val_accuracy: 0.6207
    
    Epoch 00024: val_accuracy did not improve from 0.65517
    Epoch 25/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2396 - accuracy: 0.9106 - val_loss: 1.0057 - val_accuracy: 0.5948
    
    Epoch 00025: val_accuracy did not improve from 0.65517
    Epoch 26/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2144 - accuracy: 0.9176 - val_loss: 1.0426 - val_accuracy: 0.6552
    
    Epoch 00026: val_accuracy did not improve from 0.65517
    Epoch 27/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2349 - accuracy: 0.8918 - val_loss: 1.0087 - val_accuracy: 0.5690
    
    Epoch 00027: val_accuracy did not improve from 0.65517
    Epoch 28/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2600 - accuracy: 0.9129 - val_loss: 0.9360 - val_accuracy: 0.6379
    
    Epoch 00028: val_accuracy did not improve from 0.65517
    Epoch 29/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2334 - accuracy: 0.9318 - val_loss: 1.1772 - val_accuracy: 0.5603
    
    Epoch 00029: val_accuracy did not improve from 0.65517
    Epoch 30/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2301 - accuracy: 0.9271 - val_loss: 1.0115 - val_accuracy: 0.6379
    
    Epoch 00030: val_accuracy did not improve from 0.65517
    Epoch 31/100
    5/5 [==============================] - 6s 1s/step - loss: 0.1909 - accuracy: 0.9271 - val_loss: 0.6152 - val_accuracy: 0.6379
    
    Epoch 00031: val_accuracy did not improve from 0.65517
    Epoch 32/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2042 - accuracy: 0.9247 - val_loss: 0.7155 - val_accuracy: 0.5776
    
    Epoch 00032: val_accuracy did not improve from 0.65517
    Epoch 33/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2325 - accuracy: 0.9129 - val_loss: 1.1045 - val_accuracy: 0.5776
    
    Epoch 00033: val_accuracy did not improve from 0.65517
    Epoch 34/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2080 - accuracy: 0.9200 - val_loss: 0.8210 - val_accuracy: 0.5948
    
    Epoch 00034: val_accuracy did not improve from 0.65517
    Epoch 35/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2181 - accuracy: 0.9388 - val_loss: 1.3431 - val_accuracy: 0.6466
    
    Epoch 00035: val_accuracy did not improve from 0.65517
    Epoch 36/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1822 - accuracy: 0.9435 - val_loss: 0.5970 - val_accuracy: 0.5603
    
    Epoch 00036: val_accuracy did not improve from 0.65517
    Epoch 37/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1698 - accuracy: 0.9435 - val_loss: 0.6568 - val_accuracy: 0.5776
    
    Epoch 00037: val_accuracy did not improve from 0.65517
    Epoch 38/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1941 - accuracy: 0.9341 - val_loss: 0.7269 - val_accuracy: 0.6466
    
    Epoch 00038: val_accuracy did not improve from 0.65517
    Epoch 39/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1776 - accuracy: 0.9506 - val_loss: 0.9433 - val_accuracy: 0.5517
    
    Epoch 00039: val_accuracy did not improve from 0.65517
    Epoch 40/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2145 - accuracy: 0.9176 - val_loss: 0.8426 - val_accuracy: 0.5948
    
    Epoch 00040: val_accuracy did not improve from 0.65517
    Epoch 41/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2007 - accuracy: 0.9318 - val_loss: 0.8388 - val_accuracy: 0.6293
    
    Epoch 00041: val_accuracy did not improve from 0.65517
    Epoch 42/100
    5/5 [==============================] - 6s 1s/step - loss: 0.1955 - accuracy: 0.9388 - val_loss: 0.7960 - val_accuracy: 0.6034
    
    Epoch 00042: val_accuracy did not improve from 0.65517
    Epoch 43/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2304 - accuracy: 0.9106 - val_loss: 0.9224 - val_accuracy: 0.6293
    
    Epoch 00043: val_accuracy did not improve from 0.65517
    Epoch 44/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2049 - accuracy: 0.9365 - val_loss: 1.0078 - val_accuracy: 0.6121
    
    Epoch 00044: val_accuracy did not improve from 0.65517
    Epoch 45/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2516 - accuracy: 0.8988 - val_loss: 1.3241 - val_accuracy: 0.5345
    
    Epoch 00045: val_accuracy did not improve from 0.65517
    Epoch 46/100
    5/5 [==============================] - 6s 1s/step - loss: 0.1840 - accuracy: 0.9365 - val_loss: 0.7512 - val_accuracy: 0.6293
    
    Epoch 00046: val_accuracy did not improve from 0.65517
    Epoch 47/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2076 - accuracy: 0.9294 - val_loss: 0.6247 - val_accuracy: 0.5603
    
    Epoch 00047: val_accuracy did not improve from 0.65517
    Epoch 48/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1809 - accuracy: 0.9388 - val_loss: 0.7686 - val_accuracy: 0.5776
    
    Epoch 00048: val_accuracy did not improve from 0.65517
    Epoch 49/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1735 - accuracy: 0.9529 - val_loss: 0.7904 - val_accuracy: 0.6034
    
    Epoch 00049: val_accuracy did not improve from 0.65517
    Epoch 50/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1489 - accuracy: 0.9553 - val_loss: 0.5516 - val_accuracy: 0.5603
    
    Epoch 00050: val_accuracy did not improve from 0.65517
    Epoch 51/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1657 - accuracy: 0.9459 - val_loss: 0.7497 - val_accuracy: 0.6207
    
    Epoch 00051: val_accuracy did not improve from 0.65517
    Epoch 52/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2085 - accuracy: 0.9341 - val_loss: 0.8273 - val_accuracy: 0.6034
    
    Epoch 00052: val_accuracy did not improve from 0.65517
    Epoch 53/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1620 - accuracy: 0.9576 - val_loss: 0.8747 - val_accuracy: 0.6121
    
    Epoch 00053: val_accuracy did not improve from 0.65517
    Epoch 54/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1483 - accuracy: 0.9671 - val_loss: 1.0199 - val_accuracy: 0.6379
    
    Epoch 00054: val_accuracy did not improve from 0.65517
    Epoch 55/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1832 - accuracy: 0.9412 - val_loss: 0.6383 - val_accuracy: 0.6207
    
    Epoch 00055: val_accuracy did not improve from 0.65517
    Epoch 56/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1482 - accuracy: 0.9553 - val_loss: 0.8168 - val_accuracy: 0.6034
    
    Epoch 00056: val_accuracy did not improve from 0.65517
    Epoch 57/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1530 - accuracy: 0.9506 - val_loss: 1.1613 - val_accuracy: 0.6466
    
    Epoch 00057: val_accuracy did not improve from 0.65517
    Epoch 58/100
    5/5 [==============================] - 6s 1s/step - loss: 0.1386 - accuracy: 0.9529 - val_loss: 0.7486 - val_accuracy: 0.6207
    
    Epoch 00058: val_accuracy did not improve from 0.65517
    Epoch 59/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1647 - accuracy: 0.9412 - val_loss: 0.8425 - val_accuracy: 0.6121
    
    Epoch 00059: val_accuracy did not improve from 0.65517
    Epoch 60/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1615 - accuracy: 0.9482 - val_loss: 1.1421 - val_accuracy: 0.6379
    
    Epoch 00060: val_accuracy did not improve from 0.65517
    Epoch 61/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1437 - accuracy: 0.9435 - val_loss: 0.5717 - val_accuracy: 0.6207
    
    Epoch 00061: val_accuracy did not improve from 0.65517
    Epoch 62/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1490 - accuracy: 0.9459 - val_loss: 0.9108 - val_accuracy: 0.5690
    
    Epoch 00062: val_accuracy did not improve from 0.65517
    Epoch 63/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1503 - accuracy: 0.9576 - val_loss: 0.8678 - val_accuracy: 0.6466
    
    Epoch 00063: val_accuracy did not improve from 0.65517
    Epoch 64/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1937 - accuracy: 0.9341 - val_loss: 0.6430 - val_accuracy: 0.6293
    
    Epoch 00064: val_accuracy did not improve from 0.65517
    Epoch 65/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1563 - accuracy: 0.9529 - val_loss: 0.6726 - val_accuracy: 0.6121
    
    Epoch 00065: val_accuracy did not improve from 0.65517
    Epoch 66/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1440 - accuracy: 0.9459 - val_loss: 0.9858 - val_accuracy: 0.6379
    
    Epoch 00066: val_accuracy did not improve from 0.65517
    Epoch 67/100
    5/5 [==============================] - 6s 1s/step - loss: 0.1682 - accuracy: 0.9600 - val_loss: 0.8209 - val_accuracy: 0.6207
    
    Epoch 00067: val_accuracy did not improve from 0.65517
    Epoch 68/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1370 - accuracy: 0.9576 - val_loss: 1.6919 - val_accuracy: 0.5862
    
    Epoch 00068: val_accuracy did not improve from 0.65517
    Epoch 69/100
    5/5 [==============================] - 6s 1s/step - loss: 0.1434 - accuracy: 0.9506 - val_loss: 0.8549 - val_accuracy: 0.6207
    
    Epoch 00069: val_accuracy did not improve from 0.65517
    Epoch 70/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1571 - accuracy: 0.9624 - val_loss: 0.8440 - val_accuracy: 0.6207
    
    Epoch 00070: val_accuracy did not improve from 0.65517
    Epoch 71/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1897 - accuracy: 0.9129 - val_loss: 1.4064 - val_accuracy: 0.5517
    
    Epoch 00071: val_accuracy did not improve from 0.65517
    Epoch 72/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1517 - accuracy: 0.9482 - val_loss: 1.2208 - val_accuracy: 0.6207
    
    Epoch 00072: val_accuracy did not improve from 0.65517
    Epoch 73/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1526 - accuracy: 0.9647 - val_loss: 1.4425 - val_accuracy: 0.5690
    
    Epoch 00073: val_accuracy did not improve from 0.65517
    Epoch 74/100
    5/5 [==============================] - 6s 1s/step - loss: 0.1544 - accuracy: 0.9271 - val_loss: 1.7830 - val_accuracy: 0.5431
    
    Epoch 00074: val_accuracy did not improve from 0.65517
    Epoch 75/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1369 - accuracy: 0.9576 - val_loss: 0.9955 - val_accuracy: 0.6293
    
    Epoch 00075: val_accuracy did not improve from 0.65517
    Epoch 76/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1559 - accuracy: 0.9506 - val_loss: 1.2401 - val_accuracy: 0.5517
    
    Epoch 00076: val_accuracy did not improve from 0.65517
    Epoch 77/100
    5/5 [==============================] - 6s 1s/step - loss: 0.1522 - accuracy: 0.9482 - val_loss: 0.5505 - val_accuracy: 0.5517
    
    Epoch 00077: val_accuracy did not improve from 0.65517
    Epoch 78/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1857 - accuracy: 0.9318 - val_loss: 1.1112 - val_accuracy: 0.6379
    
    Epoch 00078: val_accuracy did not improve from 0.65517
    Epoch 79/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1272 - accuracy: 0.9600 - val_loss: 1.3141 - val_accuracy: 0.6121
    
    Epoch 00079: val_accuracy did not improve from 0.65517
    Epoch 80/100
    5/5 [==============================] - 6s 1s/step - loss: 0.1519 - accuracy: 0.9576 - val_loss: 0.9357 - val_accuracy: 0.6207
    
    Epoch 00080: val_accuracy did not improve from 0.65517
    Epoch 81/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1663 - accuracy: 0.9529 - val_loss: 0.9339 - val_accuracy: 0.6207
    
    Epoch 00081: val_accuracy did not improve from 0.65517
    Epoch 82/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1385 - accuracy: 0.9529 - val_loss: 0.9202 - val_accuracy: 0.6121
    
    Epoch 00082: val_accuracy did not improve from 0.65517
    Epoch 83/100
    5/5 [==============================] - 6s 1s/step - loss: 0.1235 - accuracy: 0.9553 - val_loss: 1.0623 - val_accuracy: 0.5690
    
    Epoch 00083: val_accuracy did not improve from 0.65517
    Epoch 84/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1228 - accuracy: 0.9647 - val_loss: 1.2540 - val_accuracy: 0.6724
    
    Epoch 00084: val_accuracy improved from 0.65517 to 0.67241, saving model to classifier_resnet50_model.h5
    Epoch 85/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1396 - accuracy: 0.9600 - val_loss: 0.9323 - val_accuracy: 0.6207
    
    Epoch 00085: val_accuracy did not improve from 0.67241
    Epoch 86/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1179 - accuracy: 0.9576 - val_loss: 1.5104 - val_accuracy: 0.6207
    
    Epoch 00086: val_accuracy did not improve from 0.67241
    Epoch 87/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1157 - accuracy: 0.9718 - val_loss: 1.3363 - val_accuracy: 0.6207
    
    Epoch 00087: val_accuracy did not improve from 0.67241
    Epoch 88/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1107 - accuracy: 0.9718 - val_loss: 1.1689 - val_accuracy: 0.6552
    
    Epoch 00088: val_accuracy did not improve from 0.67241
    Epoch 89/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1380 - accuracy: 0.9482 - val_loss: 1.1954 - val_accuracy: 0.6379
    
    Epoch 00089: val_accuracy did not improve from 0.67241
    Epoch 90/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1108 - accuracy: 0.9718 - val_loss: 0.8433 - val_accuracy: 0.6638
    
    Epoch 00090: val_accuracy did not improve from 0.67241
    Epoch 91/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1152 - accuracy: 0.9694 - val_loss: 1.4037 - val_accuracy: 0.6034
    
    Epoch 00091: val_accuracy did not improve from 0.67241
    Epoch 92/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1701 - accuracy: 0.9388 - val_loss: 1.0697 - val_accuracy: 0.5172
    
    Epoch 00092: val_accuracy did not improve from 0.67241
    Epoch 93/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1179 - accuracy: 0.9624 - val_loss: 0.8235 - val_accuracy: 0.6638
    
    Epoch 00093: val_accuracy did not improve from 0.67241
    Epoch 94/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1530 - accuracy: 0.9482 - val_loss: 1.0128 - val_accuracy: 0.6552
    
    Epoch 00094: val_accuracy did not improve from 0.67241
    Epoch 95/100
    5/5 [==============================] - 6s 1s/step - loss: 0.1313 - accuracy: 0.9741 - val_loss: 1.1833 - val_accuracy: 0.6121
    
    Epoch 00095: val_accuracy did not improve from 0.67241
    Epoch 96/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1365 - accuracy: 0.9365 - val_loss: 0.7416 - val_accuracy: 0.5948
    
    Epoch 00096: val_accuracy did not improve from 0.67241
    Epoch 97/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1066 - accuracy: 0.9765 - val_loss: 1.8150 - val_accuracy: 0.6724
    
    Epoch 00097: val_accuracy did not improve from 0.67241
    Epoch 98/100
    5/5 [==============================] - 7s 1s/step - loss: 0.1208 - accuracy: 0.9506 - val_loss: 0.9613 - val_accuracy: 0.6724
    
    Epoch 00098: val_accuracy did not improve from 0.67241
    Epoch 99/100
    5/5 [==============================] - 7s 1s/step - loss: 0.0933 - accuracy: 0.9765 - val_loss: 1.1652 - val_accuracy: 0.6293
    
    Epoch 00099: val_accuracy did not improve from 0.67241
    Epoch 100/100
    5/5 [==============================] - 7s 1s/step - loss: 0.0949 - accuracy: 0.9741 - val_loss: 1.0370 - val_accuracy: 0.6466
    
    Epoch 00100: val_accuracy did not improve from 0.67241



```python
plot_training_loss_and_accuracy(resnet50_fit_history, 'Evolution of loss and accuracy with the number of training epochs for ResNet50', 'resnet50_loss_acc.png')
```


![png]({{ site.baseurl }}/images/covid19_udacity/output_39_0.png)



```python
densenet121_fit_history = train_model(model_densenet121, densenet121_train_generator, densenet121_validation_generator, 'classifier_densenet121_model.h5', num_epochs=num_epochs)
```

    Epoch 1/100
    5/5 [==============================] - 14s 3s/step - loss: 0.9742 - accuracy: 0.4729 - val_loss: 1.0985 - val_accuracy: 0.4914
    
    Epoch 00001: val_accuracy improved from -inf to 0.49138, saving model to classifier_densenet121_model.h5
    Epoch 2/100
    5/5 [==============================] - 3s 515ms/step - loss: 0.8444 - accuracy: 0.5200 - val_loss: 0.8308 - val_accuracy: 0.5259
    
    Epoch 00002: val_accuracy improved from 0.49138 to 0.52586, saving model to classifier_densenet121_model.h5
    Epoch 3/100
    5/5 [==============================] - 6s 1s/step - loss: 0.7759 - accuracy: 0.5012 - val_loss: 0.8351 - val_accuracy: 0.5172
    
    Epoch 00003: val_accuracy did not improve from 0.52586
    Epoch 4/100
    5/5 [==============================] - 7s 1s/step - loss: 0.7500 - accuracy: 0.5671 - val_loss: 1.1836 - val_accuracy: 0.5086
    
    Epoch 00004: val_accuracy did not improve from 0.52586
    Epoch 5/100
    5/5 [==============================] - 7s 1s/step - loss: 0.7000 - accuracy: 0.6306 - val_loss: 0.6780 - val_accuracy: 0.5086
    
    Epoch 00005: val_accuracy did not improve from 0.52586
    Epoch 6/100
    5/5 [==============================] - 7s 1s/step - loss: 0.6593 - accuracy: 0.6447 - val_loss: 0.8978 - val_accuracy: 0.5259
    
    Epoch 00006: val_accuracy did not improve from 0.52586
    Epoch 7/100
    5/5 [==============================] - 7s 1s/step - loss: 0.6345 - accuracy: 0.6494 - val_loss: 0.8213 - val_accuracy: 0.5086
    
    Epoch 00007: val_accuracy did not improve from 0.52586
    Epoch 8/100
    5/5 [==============================] - 6s 1s/step - loss: 0.5996 - accuracy: 0.6776 - val_loss: 0.7175 - val_accuracy: 0.5259
    
    Epoch 00008: val_accuracy did not improve from 0.52586
    Epoch 9/100
    5/5 [==============================] - 7s 1s/step - loss: 0.5637 - accuracy: 0.7059 - val_loss: 0.6776 - val_accuracy: 0.5431
    
    Epoch 00009: val_accuracy improved from 0.52586 to 0.54310, saving model to classifier_densenet121_model.h5
    Epoch 10/100
    5/5 [==============================] - 6s 1s/step - loss: 0.5316 - accuracy: 0.7106 - val_loss: 0.8260 - val_accuracy: 0.5431
    
    Epoch 00010: val_accuracy did not improve from 0.54310
    Epoch 11/100
    5/5 [==============================] - 7s 1s/step - loss: 0.5163 - accuracy: 0.7459 - val_loss: 0.9085 - val_accuracy: 0.5345
    
    Epoch 00011: val_accuracy did not improve from 0.54310
    Epoch 12/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4968 - accuracy: 0.7694 - val_loss: 0.7367 - val_accuracy: 0.5517
    
    Epoch 00012: val_accuracy improved from 0.54310 to 0.55172, saving model to classifier_densenet121_model.h5
    Epoch 13/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4922 - accuracy: 0.7624 - val_loss: 0.9852 - val_accuracy: 0.5603
    
    Epoch 00013: val_accuracy improved from 0.55172 to 0.56034, saving model to classifier_densenet121_model.h5
    Epoch 14/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4904 - accuracy: 0.7741 - val_loss: 0.9220 - val_accuracy: 0.5603
    
    Epoch 00014: val_accuracy did not improve from 0.56034
    Epoch 15/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4805 - accuracy: 0.7624 - val_loss: 0.7865 - val_accuracy: 0.5431
    
    Epoch 00015: val_accuracy did not improve from 0.56034
    Epoch 16/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4858 - accuracy: 0.7529 - val_loss: 0.6464 - val_accuracy: 0.5517
    
    Epoch 00016: val_accuracy did not improve from 0.56034
    Epoch 17/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4574 - accuracy: 0.7694 - val_loss: 1.0749 - val_accuracy: 0.5603
    
    Epoch 00017: val_accuracy did not improve from 0.56034
    Epoch 18/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4461 - accuracy: 0.8024 - val_loss: 0.7861 - val_accuracy: 0.5603
    
    Epoch 00018: val_accuracy did not improve from 0.56034
    Epoch 19/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4282 - accuracy: 0.8047 - val_loss: 0.7063 - val_accuracy: 0.5603
    
    Epoch 00019: val_accuracy did not improve from 0.56034
    Epoch 20/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4333 - accuracy: 0.7812 - val_loss: 1.0118 - val_accuracy: 0.5603
    
    Epoch 00020: val_accuracy did not improve from 0.56034
    Epoch 21/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4510 - accuracy: 0.8024 - val_loss: 0.8620 - val_accuracy: 0.5690
    
    Epoch 00021: val_accuracy improved from 0.56034 to 0.56897, saving model to classifier_densenet121_model.h5
    Epoch 22/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4713 - accuracy: 0.7765 - val_loss: 0.5762 - val_accuracy: 0.5517
    
    Epoch 00022: val_accuracy did not improve from 0.56897
    Epoch 23/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4213 - accuracy: 0.7976 - val_loss: 0.8501 - val_accuracy: 0.5517
    
    Epoch 00023: val_accuracy did not improve from 0.56897
    Epoch 24/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4043 - accuracy: 0.8188 - val_loss: 0.8414 - val_accuracy: 0.5690
    
    Epoch 00024: val_accuracy did not improve from 0.56897
    Epoch 25/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4006 - accuracy: 0.8235 - val_loss: 0.5735 - val_accuracy: 0.5603
    
    Epoch 00025: val_accuracy did not improve from 0.56897
    Epoch 26/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3818 - accuracy: 0.8235 - val_loss: 1.1220 - val_accuracy: 0.5776
    
    Epoch 00026: val_accuracy improved from 0.56897 to 0.57759, saving model to classifier_densenet121_model.h5
    Epoch 27/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3899 - accuracy: 0.8188 - val_loss: 1.0069 - val_accuracy: 0.5776
    
    Epoch 00027: val_accuracy did not improve from 0.57759
    Epoch 28/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3940 - accuracy: 0.8424 - val_loss: 1.0353 - val_accuracy: 0.5259
    
    Epoch 00028: val_accuracy did not improve from 0.57759
    Epoch 29/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4011 - accuracy: 0.8165 - val_loss: 0.5038 - val_accuracy: 0.5862
    
    Epoch 00029: val_accuracy improved from 0.57759 to 0.58621, saving model to classifier_densenet121_model.h5
    Epoch 30/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4132 - accuracy: 0.8188 - val_loss: 0.9356 - val_accuracy: 0.5862
    
    Epoch 00030: val_accuracy did not improve from 0.58621
    Epoch 31/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3846 - accuracy: 0.8329 - val_loss: 0.6371 - val_accuracy: 0.5948
    
    Epoch 00031: val_accuracy improved from 0.58621 to 0.59483, saving model to classifier_densenet121_model.h5
    Epoch 32/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3757 - accuracy: 0.8541 - val_loss: 0.8259 - val_accuracy: 0.6034
    
    Epoch 00032: val_accuracy improved from 0.59483 to 0.60345, saving model to classifier_densenet121_model.h5
    Epoch 33/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3558 - accuracy: 0.8494 - val_loss: 0.7206 - val_accuracy: 0.5776
    
    Epoch 00033: val_accuracy did not improve from 0.60345
    Epoch 34/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3517 - accuracy: 0.8471 - val_loss: 0.9660 - val_accuracy: 0.5345
    
    Epoch 00034: val_accuracy did not improve from 0.60345
    Epoch 35/100
    5/5 [==============================] - 7s 1s/step - loss: 0.4031 - accuracy: 0.8306 - val_loss: 1.0207 - val_accuracy: 0.5862
    
    Epoch 00035: val_accuracy did not improve from 0.60345
    Epoch 36/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3682 - accuracy: 0.8376 - val_loss: 0.8747 - val_accuracy: 0.5862
    
    Epoch 00036: val_accuracy did not improve from 0.60345
    Epoch 37/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3245 - accuracy: 0.8518 - val_loss: 0.9444 - val_accuracy: 0.5259
    
    Epoch 00037: val_accuracy did not improve from 0.60345
    Epoch 38/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3270 - accuracy: 0.8706 - val_loss: 0.7849 - val_accuracy: 0.5862
    
    Epoch 00038: val_accuracy did not improve from 0.60345
    Epoch 39/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3517 - accuracy: 0.8541 - val_loss: 0.7916 - val_accuracy: 0.5776
    
    Epoch 00039: val_accuracy did not improve from 0.60345
    Epoch 40/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3538 - accuracy: 0.8588 - val_loss: 0.8678 - val_accuracy: 0.5345
    
    Epoch 00040: val_accuracy did not improve from 0.60345
    Epoch 41/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3695 - accuracy: 0.8635 - val_loss: 0.5236 - val_accuracy: 0.5690
    
    Epoch 00041: val_accuracy did not improve from 0.60345
    Epoch 42/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3447 - accuracy: 0.8447 - val_loss: 0.6978 - val_accuracy: 0.5862
    
    Epoch 00042: val_accuracy did not improve from 0.60345
    Epoch 43/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3293 - accuracy: 0.8612 - val_loss: 0.5919 - val_accuracy: 0.5690
    
    Epoch 00043: val_accuracy did not improve from 0.60345
    Epoch 44/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3340 - accuracy: 0.8471 - val_loss: 0.7277 - val_accuracy: 0.5431
    
    Epoch 00044: val_accuracy did not improve from 0.60345
    Epoch 45/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3620 - accuracy: 0.8329 - val_loss: 0.9224 - val_accuracy: 0.5517
    
    Epoch 00045: val_accuracy did not improve from 0.60345
    Epoch 46/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3136 - accuracy: 0.8894 - val_loss: 0.9361 - val_accuracy: 0.5690
    
    Epoch 00046: val_accuracy did not improve from 0.60345
    Epoch 47/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3142 - accuracy: 0.8706 - val_loss: 0.9350 - val_accuracy: 0.5517
    
    Epoch 00047: val_accuracy did not improve from 0.60345
    Epoch 48/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3587 - accuracy: 0.8447 - val_loss: 0.8248 - val_accuracy: 0.5690
    
    Epoch 00048: val_accuracy did not improve from 0.60345
    Epoch 49/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3115 - accuracy: 0.8706 - val_loss: 0.7538 - val_accuracy: 0.6034
    
    Epoch 00049: val_accuracy did not improve from 0.60345
    Epoch 50/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2849 - accuracy: 0.8847 - val_loss: 0.4919 - val_accuracy: 0.5431
    
    Epoch 00050: val_accuracy did not improve from 0.60345
    Epoch 51/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3336 - accuracy: 0.8471 - val_loss: 1.1917 - val_accuracy: 0.5517
    
    Epoch 00051: val_accuracy did not improve from 0.60345
    Epoch 52/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3153 - accuracy: 0.8612 - val_loss: 0.9489 - val_accuracy: 0.6034
    
    Epoch 00052: val_accuracy did not improve from 0.60345
    Epoch 53/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3058 - accuracy: 0.8847 - val_loss: 1.0114 - val_accuracy: 0.5690
    
    Epoch 00053: val_accuracy did not improve from 0.60345
    Epoch 54/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3146 - accuracy: 0.8729 - val_loss: 0.7153 - val_accuracy: 0.5690
    
    Epoch 00054: val_accuracy did not improve from 0.60345
    Epoch 55/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3122 - accuracy: 0.8612 - val_loss: 0.7491 - val_accuracy: 0.5862
    
    Epoch 00055: val_accuracy did not improve from 0.60345
    Epoch 56/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2948 - accuracy: 0.8706 - val_loss: 0.9575 - val_accuracy: 0.5690
    
    Epoch 00056: val_accuracy did not improve from 0.60345
    Epoch 57/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3428 - accuracy: 0.8612 - val_loss: 0.6690 - val_accuracy: 0.6034
    
    Epoch 00057: val_accuracy did not improve from 0.60345
    Epoch 58/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3034 - accuracy: 0.8635 - val_loss: 0.6941 - val_accuracy: 0.6121
    
    Epoch 00058: val_accuracy improved from 0.60345 to 0.61207, saving model to classifier_densenet121_model.h5
    Epoch 59/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2999 - accuracy: 0.8800 - val_loss: 0.7340 - val_accuracy: 0.5948
    
    Epoch 00059: val_accuracy did not improve from 0.61207
    Epoch 60/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3354 - accuracy: 0.8776 - val_loss: 1.0992 - val_accuracy: 0.6034
    
    Epoch 00060: val_accuracy did not improve from 0.61207
    Epoch 61/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2985 - accuracy: 0.8918 - val_loss: 0.7360 - val_accuracy: 0.6034
    
    Epoch 00061: val_accuracy did not improve from 0.61207
    Epoch 62/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3047 - accuracy: 0.8682 - val_loss: 0.8439 - val_accuracy: 0.5948
    
    Epoch 00062: val_accuracy did not improve from 0.61207
    Epoch 63/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3246 - accuracy: 0.8400 - val_loss: 0.9834 - val_accuracy: 0.6121
    
    Epoch 00063: val_accuracy did not improve from 0.61207
    Epoch 64/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2961 - accuracy: 0.8800 - val_loss: 0.6838 - val_accuracy: 0.5690
    
    Epoch 00064: val_accuracy did not improve from 0.61207
    Epoch 65/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3633 - accuracy: 0.8494 - val_loss: 0.8774 - val_accuracy: 0.6207
    
    Epoch 00065: val_accuracy improved from 0.61207 to 0.62069, saving model to classifier_densenet121_model.h5
    Epoch 66/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2971 - accuracy: 0.8753 - val_loss: 0.9768 - val_accuracy: 0.6207
    
    Epoch 00066: val_accuracy did not improve from 0.62069
    Epoch 67/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2931 - accuracy: 0.8729 - val_loss: 0.8421 - val_accuracy: 0.5948
    
    Epoch 00067: val_accuracy did not improve from 0.62069
    Epoch 68/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2581 - accuracy: 0.9035 - val_loss: 0.8495 - val_accuracy: 0.6121
    
    Epoch 00068: val_accuracy did not improve from 0.62069
    Epoch 69/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2698 - accuracy: 0.8988 - val_loss: 0.7111 - val_accuracy: 0.6121
    
    Epoch 00069: val_accuracy did not improve from 0.62069
    Epoch 70/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2857 - accuracy: 0.8682 - val_loss: 0.6502 - val_accuracy: 0.6121
    
    Epoch 00070: val_accuracy did not improve from 0.62069
    Epoch 71/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2759 - accuracy: 0.8824 - val_loss: 0.7819 - val_accuracy: 0.5948
    
    Epoch 00071: val_accuracy did not improve from 0.62069
    Epoch 72/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2700 - accuracy: 0.8871 - val_loss: 0.6703 - val_accuracy: 0.6293
    
    Epoch 00072: val_accuracy improved from 0.62069 to 0.62931, saving model to classifier_densenet121_model.h5
    Epoch 73/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2464 - accuracy: 0.9200 - val_loss: 0.8504 - val_accuracy: 0.6121
    
    Epoch 00073: val_accuracy did not improve from 0.62931
    Epoch 74/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2757 - accuracy: 0.8988 - val_loss: 0.7740 - val_accuracy: 0.6034
    
    Epoch 00074: val_accuracy did not improve from 0.62931
    Epoch 75/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2712 - accuracy: 0.9035 - val_loss: 0.7279 - val_accuracy: 0.6207
    
    Epoch 00075: val_accuracy did not improve from 0.62931
    Epoch 76/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2786 - accuracy: 0.8965 - val_loss: 0.7634 - val_accuracy: 0.6379
    
    Epoch 00076: val_accuracy improved from 0.62931 to 0.63793, saving model to classifier_densenet121_model.h5
    Epoch 77/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2530 - accuracy: 0.9012 - val_loss: 0.7016 - val_accuracy: 0.6121
    
    Epoch 00077: val_accuracy did not improve from 0.63793
    Epoch 78/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3109 - accuracy: 0.8776 - val_loss: 0.8219 - val_accuracy: 0.6379
    
    Epoch 00078: val_accuracy did not improve from 0.63793
    Epoch 79/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2521 - accuracy: 0.9153 - val_loss: 0.9657 - val_accuracy: 0.6121
    
    Epoch 00079: val_accuracy did not improve from 0.63793
    Epoch 80/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2829 - accuracy: 0.8776 - val_loss: 0.5277 - val_accuracy: 0.6207
    
    Epoch 00080: val_accuracy did not improve from 0.63793
    Epoch 81/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2615 - accuracy: 0.8776 - val_loss: 0.6180 - val_accuracy: 0.6379
    
    Epoch 00081: val_accuracy did not improve from 0.63793
    Epoch 82/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2661 - accuracy: 0.8918 - val_loss: 0.5360 - val_accuracy: 0.6034
    
    Epoch 00082: val_accuracy did not improve from 0.63793
    Epoch 83/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2853 - accuracy: 0.8894 - val_loss: 0.6588 - val_accuracy: 0.6466
    
    Epoch 00083: val_accuracy improved from 0.63793 to 0.64655, saving model to classifier_densenet121_model.h5
    Epoch 84/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2795 - accuracy: 0.8824 - val_loss: 0.6061 - val_accuracy: 0.6034
    
    Epoch 00084: val_accuracy did not improve from 0.64655
    Epoch 85/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2790 - accuracy: 0.8965 - val_loss: 1.0799 - val_accuracy: 0.6034
    
    Epoch 00085: val_accuracy did not improve from 0.64655
    Epoch 86/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2841 - accuracy: 0.8682 - val_loss: 0.6812 - val_accuracy: 0.6379
    
    Epoch 00086: val_accuracy did not improve from 0.64655
    Epoch 87/100
    5/5 [==============================] - 7s 1s/step - loss: 0.3111 - accuracy: 0.8824 - val_loss: 0.9518 - val_accuracy: 0.5948
    
    Epoch 00087: val_accuracy did not improve from 0.64655
    Epoch 88/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2697 - accuracy: 0.8871 - val_loss: 0.6648 - val_accuracy: 0.6466
    
    Epoch 00088: val_accuracy did not improve from 0.64655
    Epoch 89/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2964 - accuracy: 0.8894 - val_loss: 0.7203 - val_accuracy: 0.6379
    
    Epoch 00089: val_accuracy did not improve from 0.64655
    Epoch 90/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2734 - accuracy: 0.8847 - val_loss: 1.0855 - val_accuracy: 0.6034
    
    Epoch 00090: val_accuracy did not improve from 0.64655
    Epoch 91/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2788 - accuracy: 0.8682 - val_loss: 0.8294 - val_accuracy: 0.5862
    
    Epoch 00091: val_accuracy did not improve from 0.64655
    Epoch 92/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2884 - accuracy: 0.8753 - val_loss: 0.8920 - val_accuracy: 0.6207
    
    Epoch 00092: val_accuracy did not improve from 0.64655
    Epoch 93/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2463 - accuracy: 0.8941 - val_loss: 0.5384 - val_accuracy: 0.6379
    
    Epoch 00093: val_accuracy did not improve from 0.64655
    Epoch 94/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2587 - accuracy: 0.8941 - val_loss: 0.7990 - val_accuracy: 0.6207
    
    Epoch 00094: val_accuracy did not improve from 0.64655
    Epoch 95/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2578 - accuracy: 0.9059 - val_loss: 0.7435 - val_accuracy: 0.6034
    
    Epoch 00095: val_accuracy did not improve from 0.64655
    Epoch 96/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2482 - accuracy: 0.8894 - val_loss: 0.7701 - val_accuracy: 0.6207
    
    Epoch 00096: val_accuracy did not improve from 0.64655
    Epoch 97/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2287 - accuracy: 0.8988 - val_loss: 0.6136 - val_accuracy: 0.6552
    
    Epoch 00097: val_accuracy improved from 0.64655 to 0.65517, saving model to classifier_densenet121_model.h5
    Epoch 98/100
    5/5 [==============================] - 6s 1s/step - loss: 0.2465 - accuracy: 0.9200 - val_loss: 0.6678 - val_accuracy: 0.6379
    
    Epoch 00098: val_accuracy did not improve from 0.65517
    Epoch 99/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2284 - accuracy: 0.9176 - val_loss: 0.9820 - val_accuracy: 0.6207
    
    Epoch 00099: val_accuracy did not improve from 0.65517
    Epoch 100/100
    5/5 [==============================] - 7s 1s/step - loss: 0.2498 - accuracy: 0.8871 - val_loss: 0.7006 - val_accuracy: 0.6379
    
    Epoch 00100: val_accuracy did not improve from 0.65517



```python
plot_training_loss_and_accuracy(densenet121_fit_history, 'Evolution of loss and accuracy with the number of training epochs for DenseNet121', 'densenet121_loss_acc.png')
```


![png]({{ site.baseurl }}/images/covid19_udacity/output_41_0.png)



```python
nasnetmobile_fit_history = train_model(model_nasnetmobile, nasnetmobile_train_generator, nasnetmobile_validation_generator, 'classifier_nasnetmobile_model.h5', num_epochs=num_epochs)
```

    Epoch 1/100
    5/5 [==============================] - 15s 3s/step - loss: 0.7665 - accuracy: 0.5576 - val_loss: 0.6967 - val_accuracy: 0.5345
    
    Epoch 00001: val_accuracy improved from -inf to 0.53448, saving model to classifier_nasnetmobile_model.h5
    Epoch 2/100
    5/5 [==============================] - 2s 325ms/step - loss: 0.6546 - accuracy: 0.6424 - val_loss: 0.7449 - val_accuracy: 0.4914
    
    Epoch 00002: val_accuracy did not improve from 0.53448
    Epoch 3/100
    5/5 [==============================] - 6s 1s/step - loss: 0.6234 - accuracy: 0.6259 - val_loss: 0.7134 - val_accuracy: 0.5259
    
    Epoch 00003: val_accuracy did not improve from 0.53448
    Epoch 4/100
    5/5 [==============================] - 6s 1s/step - loss: 0.6008 - accuracy: 0.7129 - val_loss: 0.8670 - val_accuracy: 0.5603
    
    Epoch 00004: val_accuracy improved from 0.53448 to 0.56034, saving model to classifier_nasnetmobile_model.h5
    Epoch 5/100
    5/5 [==============================] - 6s 1s/step - loss: 0.5404 - accuracy: 0.7412 - val_loss: 0.7654 - val_accuracy: 0.5259
    
    Epoch 00005: val_accuracy did not improve from 0.56034
    Epoch 6/100
    5/5 [==============================] - 6s 1s/step - loss: 0.6086 - accuracy: 0.6988 - val_loss: 0.7551 - val_accuracy: 0.5431
    
    Epoch 00006: val_accuracy did not improve from 0.56034
    Epoch 7/100
    5/5 [==============================] - 6s 1s/step - loss: 0.5555 - accuracy: 0.7365 - val_loss: 1.0706 - val_accuracy: 0.5517
    
    Epoch 00007: val_accuracy did not improve from 0.56034
    Epoch 8/100
    5/5 [==============================] - 6s 1s/step - loss: 0.5248 - accuracy: 0.7271 - val_loss: 0.6823 - val_accuracy: 0.5345
    
    Epoch 00008: val_accuracy did not improve from 0.56034
    Epoch 9/100
    5/5 [==============================] - 6s 1s/step - loss: 0.5220 - accuracy: 0.7600 - val_loss: 1.0582 - val_accuracy: 0.5345
    
    Epoch 00009: val_accuracy did not improve from 0.56034
    Epoch 10/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4938 - accuracy: 0.7553 - val_loss: 0.7058 - val_accuracy: 0.5345
    
    Epoch 00010: val_accuracy did not improve from 0.56034
    Epoch 11/100
    5/5 [==============================] - 6s 1s/step - loss: 0.5086 - accuracy: 0.7459 - val_loss: 0.8373 - val_accuracy: 0.5690
    
    Epoch 00011: val_accuracy improved from 0.56034 to 0.56897, saving model to classifier_nasnetmobile_model.h5
    Epoch 12/100
    5/5 [==============================] - 6s 1s/step - loss: 0.5050 - accuracy: 0.7459 - val_loss: 0.7473 - val_accuracy: 0.5431
    
    Epoch 00012: val_accuracy did not improve from 0.56897
    Epoch 13/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4701 - accuracy: 0.7906 - val_loss: 0.8287 - val_accuracy: 0.5431
    
    Epoch 00013: val_accuracy did not improve from 0.56897
    Epoch 14/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4700 - accuracy: 0.7765 - val_loss: 0.7765 - val_accuracy: 0.5862
    
    Epoch 00014: val_accuracy improved from 0.56897 to 0.58621, saving model to classifier_nasnetmobile_model.h5
    Epoch 15/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4809 - accuracy: 0.7553 - val_loss: 0.7329 - val_accuracy: 0.5862
    
    Epoch 00015: val_accuracy did not improve from 0.58621
    Epoch 16/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4573 - accuracy: 0.7882 - val_loss: 0.8275 - val_accuracy: 0.5603
    
    Epoch 00016: val_accuracy did not improve from 0.58621
    Epoch 17/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4494 - accuracy: 0.7741 - val_loss: 0.7070 - val_accuracy: 0.5431
    
    Epoch 00017: val_accuracy did not improve from 0.58621
    Epoch 18/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4369 - accuracy: 0.8071 - val_loss: 0.6723 - val_accuracy: 0.5776
    
    Epoch 00018: val_accuracy did not improve from 0.58621
    Epoch 19/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4794 - accuracy: 0.8094 - val_loss: 0.7521 - val_accuracy: 0.5690
    
    Epoch 00019: val_accuracy did not improve from 0.58621
    Epoch 20/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4416 - accuracy: 0.7882 - val_loss: 0.6318 - val_accuracy: 0.5776
    
    Epoch 00020: val_accuracy did not improve from 0.58621
    Epoch 21/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4353 - accuracy: 0.7976 - val_loss: 0.6158 - val_accuracy: 0.5345
    
    Epoch 00021: val_accuracy did not improve from 0.58621
    Epoch 22/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4783 - accuracy: 0.7624 - val_loss: 0.6480 - val_accuracy: 0.5862
    
    Epoch 00022: val_accuracy did not improve from 0.58621
    Epoch 23/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4327 - accuracy: 0.8071 - val_loss: 0.6612 - val_accuracy: 0.5776
    
    Epoch 00023: val_accuracy did not improve from 0.58621
    Epoch 24/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4647 - accuracy: 0.7859 - val_loss: 0.6203 - val_accuracy: 0.6034
    
    Epoch 00024: val_accuracy improved from 0.58621 to 0.60345, saving model to classifier_nasnetmobile_model.h5
    Epoch 25/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4366 - accuracy: 0.7906 - val_loss: 0.7752 - val_accuracy: 0.5862
    
    Epoch 00025: val_accuracy did not improve from 0.60345
    Epoch 26/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4431 - accuracy: 0.8000 - val_loss: 0.8234 - val_accuracy: 0.6121
    
    Epoch 00026: val_accuracy improved from 0.60345 to 0.61207, saving model to classifier_nasnetmobile_model.h5
    Epoch 27/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4272 - accuracy: 0.8000 - val_loss: 0.9225 - val_accuracy: 0.6293
    
    Epoch 00027: val_accuracy improved from 0.61207 to 0.62931, saving model to classifier_nasnetmobile_model.h5
    Epoch 28/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4346 - accuracy: 0.8047 - val_loss: 0.8398 - val_accuracy: 0.6207
    
    Epoch 00028: val_accuracy did not improve from 0.62931
    Epoch 29/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4163 - accuracy: 0.7929 - val_loss: 1.1049 - val_accuracy: 0.5948
    
    Epoch 00029: val_accuracy did not improve from 0.62931
    Epoch 30/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4072 - accuracy: 0.8141 - val_loss: 0.8296 - val_accuracy: 0.6207
    
    Epoch 00030: val_accuracy did not improve from 0.62931
    Epoch 31/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4000 - accuracy: 0.8071 - val_loss: 0.7526 - val_accuracy: 0.6293
    
    Epoch 00031: val_accuracy did not improve from 0.62931
    Epoch 32/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3959 - accuracy: 0.8306 - val_loss: 0.6454 - val_accuracy: 0.6293
    
    Epoch 00032: val_accuracy did not improve from 0.62931
    Epoch 33/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4022 - accuracy: 0.8118 - val_loss: 0.9706 - val_accuracy: 0.6293
    
    Epoch 00033: val_accuracy did not improve from 0.62931
    Epoch 34/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3972 - accuracy: 0.7906 - val_loss: 0.7270 - val_accuracy: 0.6207
    
    Epoch 00034: val_accuracy did not improve from 0.62931
    Epoch 35/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3774 - accuracy: 0.8306 - val_loss: 0.9551 - val_accuracy: 0.6293
    
    Epoch 00035: val_accuracy did not improve from 0.62931
    Epoch 36/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3972 - accuracy: 0.8212 - val_loss: 0.8610 - val_accuracy: 0.5862
    
    Epoch 00036: val_accuracy did not improve from 0.62931
    Epoch 37/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3721 - accuracy: 0.8235 - val_loss: 0.6459 - val_accuracy: 0.6121
    
    Epoch 00037: val_accuracy did not improve from 0.62931
    Epoch 38/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4029 - accuracy: 0.8329 - val_loss: 0.7406 - val_accuracy: 0.5948
    
    Epoch 00038: val_accuracy did not improve from 0.62931
    Epoch 39/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3882 - accuracy: 0.8259 - val_loss: 0.4169 - val_accuracy: 0.6034
    
    Epoch 00039: val_accuracy did not improve from 0.62931
    Epoch 40/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3917 - accuracy: 0.7976 - val_loss: 0.9224 - val_accuracy: 0.6293
    
    Epoch 00040: val_accuracy did not improve from 0.62931
    Epoch 41/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4147 - accuracy: 0.8141 - val_loss: 0.7138 - val_accuracy: 0.6379
    
    Epoch 00041: val_accuracy improved from 0.62931 to 0.63793, saving model to classifier_nasnetmobile_model.h5
    Epoch 42/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3517 - accuracy: 0.8329 - val_loss: 0.8183 - val_accuracy: 0.5690
    
    Epoch 00042: val_accuracy did not improve from 0.63793
    Epoch 43/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4000 - accuracy: 0.8094 - val_loss: 0.7997 - val_accuracy: 0.5948
    
    Epoch 00043: val_accuracy did not improve from 0.63793
    Epoch 44/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3960 - accuracy: 0.8047 - val_loss: 0.7677 - val_accuracy: 0.6466
    
    Epoch 00044: val_accuracy improved from 0.63793 to 0.64655, saving model to classifier_nasnetmobile_model.h5
    Epoch 45/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4125 - accuracy: 0.8024 - val_loss: 0.6160 - val_accuracy: 0.6466
    
    Epoch 00045: val_accuracy did not improve from 0.64655
    Epoch 46/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3748 - accuracy: 0.8329 - val_loss: 0.6953 - val_accuracy: 0.5776
    
    Epoch 00046: val_accuracy did not improve from 0.64655
    Epoch 47/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3881 - accuracy: 0.8071 - val_loss: 0.6005 - val_accuracy: 0.5862
    
    Epoch 00047: val_accuracy did not improve from 0.64655
    Epoch 48/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4238 - accuracy: 0.8094 - val_loss: 0.7668 - val_accuracy: 0.6293
    
    Epoch 00048: val_accuracy did not improve from 0.64655
    Epoch 49/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4167 - accuracy: 0.8353 - val_loss: 0.5587 - val_accuracy: 0.6121
    
    Epoch 00049: val_accuracy did not improve from 0.64655
    Epoch 50/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4054 - accuracy: 0.8212 - val_loss: 0.7846 - val_accuracy: 0.6207
    
    Epoch 00050: val_accuracy did not improve from 0.64655
    Epoch 51/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3665 - accuracy: 0.8353 - val_loss: 0.9189 - val_accuracy: 0.6293
    
    Epoch 00051: val_accuracy did not improve from 0.64655
    Epoch 52/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3702 - accuracy: 0.8353 - val_loss: 0.7069 - val_accuracy: 0.5603
    
    Epoch 00052: val_accuracy did not improve from 0.64655
    Epoch 53/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3697 - accuracy: 0.8282 - val_loss: 0.6177 - val_accuracy: 0.6121
    
    Epoch 00053: val_accuracy did not improve from 0.64655
    Epoch 54/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3924 - accuracy: 0.8306 - val_loss: 1.1514 - val_accuracy: 0.6293
    
    Epoch 00054: val_accuracy did not improve from 0.64655
    Epoch 55/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3711 - accuracy: 0.8235 - val_loss: 0.5849 - val_accuracy: 0.6293
    
    Epoch 00055: val_accuracy did not improve from 0.64655
    Epoch 56/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3585 - accuracy: 0.8306 - val_loss: 0.5010 - val_accuracy: 0.6293
    
    Epoch 00056: val_accuracy did not improve from 0.64655
    Epoch 57/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3536 - accuracy: 0.8282 - val_loss: 0.6953 - val_accuracy: 0.6121
    
    Epoch 00057: val_accuracy did not improve from 0.64655
    Epoch 58/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3688 - accuracy: 0.8659 - val_loss: 0.4501 - val_accuracy: 0.6034
    
    Epoch 00058: val_accuracy did not improve from 0.64655
    Epoch 59/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3512 - accuracy: 0.8541 - val_loss: 0.9416 - val_accuracy: 0.6379
    
    Epoch 00059: val_accuracy did not improve from 0.64655
    Epoch 60/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3919 - accuracy: 0.8447 - val_loss: 0.7055 - val_accuracy: 0.6121
    
    Epoch 00060: val_accuracy did not improve from 0.64655
    Epoch 61/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3599 - accuracy: 0.8541 - val_loss: 0.7611 - val_accuracy: 0.6121
    
    Epoch 00061: val_accuracy did not improve from 0.64655
    Epoch 62/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3462 - accuracy: 0.8306 - val_loss: 1.0311 - val_accuracy: 0.6121
    
    Epoch 00062: val_accuracy did not improve from 0.64655
    Epoch 63/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3690 - accuracy: 0.8165 - val_loss: 0.6887 - val_accuracy: 0.5862
    
    Epoch 00063: val_accuracy did not improve from 0.64655
    Epoch 64/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4026 - accuracy: 0.8353 - val_loss: 0.7302 - val_accuracy: 0.6379
    
    Epoch 00064: val_accuracy did not improve from 0.64655
    Epoch 65/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3643 - accuracy: 0.8471 - val_loss: 1.0896 - val_accuracy: 0.6121
    
    Epoch 00065: val_accuracy did not improve from 0.64655
    Epoch 66/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3571 - accuracy: 0.8424 - val_loss: 0.8810 - val_accuracy: 0.6121
    
    Epoch 00066: val_accuracy did not improve from 0.64655
    Epoch 67/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4050 - accuracy: 0.8212 - val_loss: 0.5504 - val_accuracy: 0.6466
    
    Epoch 00067: val_accuracy did not improve from 0.64655
    Epoch 68/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3594 - accuracy: 0.8259 - val_loss: 0.5928 - val_accuracy: 0.6121
    
    Epoch 00068: val_accuracy did not improve from 0.64655
    Epoch 69/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3879 - accuracy: 0.8141 - val_loss: 0.5974 - val_accuracy: 0.6379
    
    Epoch 00069: val_accuracy did not improve from 0.64655
    Epoch 70/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3977 - accuracy: 0.8235 - val_loss: 0.6081 - val_accuracy: 0.5690
    
    Epoch 00070: val_accuracy did not improve from 0.64655
    Epoch 71/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3786 - accuracy: 0.8518 - val_loss: 0.8988 - val_accuracy: 0.6121
    
    Epoch 00071: val_accuracy did not improve from 0.64655
    Epoch 72/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3405 - accuracy: 0.8518 - val_loss: 0.5007 - val_accuracy: 0.6466
    
    Epoch 00072: val_accuracy did not improve from 0.64655
    Epoch 73/100
    5/5 [==============================] - 6s 1s/step - loss: 0.4070 - accuracy: 0.8282 - val_loss: 0.4072 - val_accuracy: 0.5776
    
    Epoch 00073: val_accuracy did not improve from 0.64655
    Epoch 74/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3080 - accuracy: 0.8588 - val_loss: 0.6375 - val_accuracy: 0.6034
    
    Epoch 00074: val_accuracy did not improve from 0.64655
    Epoch 75/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3751 - accuracy: 0.8047 - val_loss: 0.5971 - val_accuracy: 0.6034
    
    Epoch 00075: val_accuracy did not improve from 0.64655
    Epoch 76/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3445 - accuracy: 0.8518 - val_loss: 0.7412 - val_accuracy: 0.6121
    
    Epoch 00076: val_accuracy did not improve from 0.64655
    Epoch 77/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3626 - accuracy: 0.8565 - val_loss: 0.8074 - val_accuracy: 0.6293
    
    Epoch 00077: val_accuracy did not improve from 0.64655
    Epoch 78/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3112 - accuracy: 0.8635 - val_loss: 0.5079 - val_accuracy: 0.6121
    
    Epoch 00078: val_accuracy did not improve from 0.64655
    Epoch 79/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3604 - accuracy: 0.8494 - val_loss: 0.7083 - val_accuracy: 0.6466
    
    Epoch 00079: val_accuracy did not improve from 0.64655
    Epoch 80/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3503 - accuracy: 0.8659 - val_loss: 0.7057 - val_accuracy: 0.6121
    
    Epoch 00080: val_accuracy did not improve from 0.64655
    Epoch 81/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3168 - accuracy: 0.8541 - val_loss: 1.0031 - val_accuracy: 0.6207
    
    Epoch 00081: val_accuracy did not improve from 0.64655
    Epoch 82/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3594 - accuracy: 0.8565 - val_loss: 0.8024 - val_accuracy: 0.6207
    
    Epoch 00082: val_accuracy did not improve from 0.64655
    Epoch 83/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3693 - accuracy: 0.8376 - val_loss: 0.7321 - val_accuracy: 0.6121
    
    Epoch 00083: val_accuracy did not improve from 0.64655
    Epoch 84/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3587 - accuracy: 0.8447 - val_loss: 0.7465 - val_accuracy: 0.6293
    
    Epoch 00084: val_accuracy did not improve from 0.64655
    Epoch 85/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3492 - accuracy: 0.8329 - val_loss: 0.4137 - val_accuracy: 0.6034
    
    Epoch 00085: val_accuracy did not improve from 0.64655
    Epoch 86/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3109 - accuracy: 0.8706 - val_loss: 0.7398 - val_accuracy: 0.6034
    
    Epoch 00086: val_accuracy did not improve from 0.64655
    Epoch 87/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3266 - accuracy: 0.8776 - val_loss: 0.6228 - val_accuracy: 0.6207
    
    Epoch 00087: val_accuracy did not improve from 0.64655
    Epoch 88/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3414 - accuracy: 0.8424 - val_loss: 0.9512 - val_accuracy: 0.6121
    
    Epoch 00088: val_accuracy did not improve from 0.64655
    Epoch 89/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3347 - accuracy: 0.8588 - val_loss: 1.0701 - val_accuracy: 0.5948
    
    Epoch 00089: val_accuracy did not improve from 0.64655
    Epoch 90/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3159 - accuracy: 0.8729 - val_loss: 0.7827 - val_accuracy: 0.6293
    
    Epoch 00090: val_accuracy did not improve from 0.64655
    Epoch 91/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3372 - accuracy: 0.8447 - val_loss: 0.4071 - val_accuracy: 0.6207
    
    Epoch 00091: val_accuracy did not improve from 0.64655
    Epoch 92/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3377 - accuracy: 0.8612 - val_loss: 0.7041 - val_accuracy: 0.6034
    
    Epoch 00092: val_accuracy did not improve from 0.64655
    Epoch 93/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3316 - accuracy: 0.8400 - val_loss: 0.6161 - val_accuracy: 0.6293
    
    Epoch 00093: val_accuracy did not improve from 0.64655
    Epoch 94/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3245 - accuracy: 0.8729 - val_loss: 0.6823 - val_accuracy: 0.6121
    
    Epoch 00094: val_accuracy did not improve from 0.64655
    Epoch 95/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3357 - accuracy: 0.8471 - val_loss: 0.6354 - val_accuracy: 0.6207
    
    Epoch 00095: val_accuracy did not improve from 0.64655
    Epoch 96/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3134 - accuracy: 0.8565 - val_loss: 0.5802 - val_accuracy: 0.6207
    
    Epoch 00096: val_accuracy did not improve from 0.64655
    Epoch 97/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3236 - accuracy: 0.8776 - val_loss: 0.8915 - val_accuracy: 0.6293
    
    Epoch 00097: val_accuracy did not improve from 0.64655
    Epoch 98/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3460 - accuracy: 0.8471 - val_loss: 0.8375 - val_accuracy: 0.6207
    
    Epoch 00098: val_accuracy did not improve from 0.64655
    Epoch 99/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3156 - accuracy: 0.8729 - val_loss: 0.8949 - val_accuracy: 0.6293
    
    Epoch 00099: val_accuracy did not improve from 0.64655
    Epoch 100/100
    5/5 [==============================] - 6s 1s/step - loss: 0.3308 - accuracy: 0.8565 - val_loss: 0.9368 - val_accuracy: 0.6293
    
    Epoch 00100: val_accuracy did not improve from 0.64655



```python
plot_training_loss_and_accuracy(nasnetmobile_fit_history, 'Evolution of loss and accuracy with the number of training epochs for NASNetMobile', 'nasnetmobile_loss_acc.png')
```


![png]({{ site.baseurl }}/images/covid19_udacity/output_43_0.png)



```python
inceptionv3_fit_history = train_model(model_inceptionv3, inceptionv3_train_generator, inceptionv3_validation_generator, 'classifier_inceptionv3_model.h5', num_epochs=num_epochs)
```

    Epoch 1/100
    5/5 [==============================] - 15s 3s/step - loss: 0.6197 - accuracy: 0.6682 - val_loss: 0.8162 - val_accuracy: 0.4310
    
    Epoch 00001: val_accuracy improved from -inf to 0.43103, saving model to classifier_inceptionv3_model.h5
    Epoch 2/100
    5/5 [==============================] - 3s 576ms/step - loss: 0.5997 - accuracy: 0.7106 - val_loss: 0.6991 - val_accuracy: 0.4741
    
    Epoch 00002: val_accuracy improved from 0.43103 to 0.47414, saving model to classifier_inceptionv3_model.h5
    Epoch 3/100
    5/5 [==============================] - 9s 2s/step - loss: 0.5513 - accuracy: 0.7059 - val_loss: 0.6241 - val_accuracy: 0.5000
    
    Epoch 00003: val_accuracy improved from 0.47414 to 0.50000, saving model to classifier_inceptionv3_model.h5
    Epoch 4/100
    5/5 [==============================] - 9s 2s/step - loss: 0.5480 - accuracy: 0.7435 - val_loss: 0.7348 - val_accuracy: 0.4914
    
    Epoch 00004: val_accuracy did not improve from 0.50000
    Epoch 5/100
    5/5 [==============================] - 10s 2s/step - loss: 0.5177 - accuracy: 0.7529 - val_loss: 0.7168 - val_accuracy: 0.5776
    
    Epoch 00005: val_accuracy improved from 0.50000 to 0.57759, saving model to classifier_inceptionv3_model.h5
    Epoch 6/100
    5/5 [==============================] - 9s 2s/step - loss: 0.4743 - accuracy: 0.8094 - val_loss: 0.8305 - val_accuracy: 0.4914
    
    Epoch 00006: val_accuracy did not improve from 0.57759
    Epoch 7/100
    5/5 [==============================] - 10s 2s/step - loss: 0.4924 - accuracy: 0.7976 - val_loss: 0.6126 - val_accuracy: 0.5517
    
    Epoch 00007: val_accuracy did not improve from 0.57759
    Epoch 8/100
    5/5 [==============================] - 10s 2s/step - loss: 0.4636 - accuracy: 0.8024 - val_loss: 0.9740 - val_accuracy: 0.4914
    
    Epoch 00008: val_accuracy did not improve from 0.57759
    Epoch 9/100
    5/5 [==============================] - 10s 2s/step - loss: 0.4603 - accuracy: 0.7929 - val_loss: 0.6598 - val_accuracy: 0.5345
    
    Epoch 00009: val_accuracy did not improve from 0.57759
    Epoch 10/100
    5/5 [==============================] - 10s 2s/step - loss: 0.4441 - accuracy: 0.8094 - val_loss: 0.8151 - val_accuracy: 0.5259
    
    Epoch 00010: val_accuracy did not improve from 0.57759
    Epoch 11/100
    5/5 [==============================] - 10s 2s/step - loss: 0.4369 - accuracy: 0.8094 - val_loss: 0.7038 - val_accuracy: 0.5172
    
    Epoch 00011: val_accuracy did not improve from 0.57759
    Epoch 12/100
    5/5 [==============================] - 9s 2s/step - loss: 0.4450 - accuracy: 0.8118 - val_loss: 0.5869 - val_accuracy: 0.5345
    
    Epoch 00012: val_accuracy did not improve from 0.57759
    Epoch 13/100
    5/5 [==============================] - 10s 2s/step - loss: 0.4319 - accuracy: 0.8212 - val_loss: 0.8354 - val_accuracy: 0.5172
    
    Epoch 00013: val_accuracy did not improve from 0.57759
    Epoch 14/100
    5/5 [==============================] - 9s 2s/step - loss: 0.4105 - accuracy: 0.8259 - val_loss: 0.9379 - val_accuracy: 0.5086
    
    Epoch 00014: val_accuracy did not improve from 0.57759
    Epoch 15/100
    5/5 [==============================] - 10s 2s/step - loss: 0.4017 - accuracy: 0.8188 - val_loss: 1.0471 - val_accuracy: 0.5172
    
    Epoch 00015: val_accuracy did not improve from 0.57759
    Epoch 16/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3781 - accuracy: 0.8282 - val_loss: 0.8765 - val_accuracy: 0.5172
    
    Epoch 00016: val_accuracy did not improve from 0.57759
    Epoch 17/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3971 - accuracy: 0.8447 - val_loss: 0.7924 - val_accuracy: 0.5172
    
    Epoch 00017: val_accuracy did not improve from 0.57759
    Epoch 18/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3969 - accuracy: 0.8259 - val_loss: 0.8774 - val_accuracy: 0.5259
    
    Epoch 00018: val_accuracy did not improve from 0.57759
    Epoch 19/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3703 - accuracy: 0.8471 - val_loss: 0.7423 - val_accuracy: 0.5000
    
    Epoch 00019: val_accuracy did not improve from 0.57759
    Epoch 20/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3502 - accuracy: 0.8706 - val_loss: 1.3111 - val_accuracy: 0.5172
    
    Epoch 00020: val_accuracy did not improve from 0.57759
    Epoch 21/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3990 - accuracy: 0.8447 - val_loss: 0.7884 - val_accuracy: 0.5345
    
    Epoch 00021: val_accuracy did not improve from 0.57759
    Epoch 22/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3954 - accuracy: 0.8494 - val_loss: 0.6510 - val_accuracy: 0.5172
    
    Epoch 00022: val_accuracy did not improve from 0.57759
    Epoch 23/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3646 - accuracy: 0.8588 - val_loss: 0.6642 - val_accuracy: 0.5603
    
    Epoch 00023: val_accuracy did not improve from 0.57759
    Epoch 24/100
    5/5 [==============================] - 10s 2s/step - loss: 0.4002 - accuracy: 0.8376 - val_loss: 0.7791 - val_accuracy: 0.5345
    
    Epoch 00024: val_accuracy did not improve from 0.57759
    Epoch 25/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3570 - accuracy: 0.8541 - val_loss: 0.8102 - val_accuracy: 0.5345
    
    Epoch 00025: val_accuracy did not improve from 0.57759
    Epoch 26/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3679 - accuracy: 0.8518 - val_loss: 0.6940 - val_accuracy: 0.5517
    
    Epoch 00026: val_accuracy did not improve from 0.57759
    Epoch 27/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3820 - accuracy: 0.8165 - val_loss: 0.9501 - val_accuracy: 0.5345
    
    Epoch 00027: val_accuracy did not improve from 0.57759
    Epoch 28/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3368 - accuracy: 0.8871 - val_loss: 0.5906 - val_accuracy: 0.5862
    
    Epoch 00028: val_accuracy improved from 0.57759 to 0.58621, saving model to classifier_inceptionv3_model.h5
    Epoch 29/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3862 - accuracy: 0.8400 - val_loss: 0.6438 - val_accuracy: 0.5517
    
    Epoch 00029: val_accuracy did not improve from 0.58621
    Epoch 30/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3405 - accuracy: 0.8471 - val_loss: 0.8059 - val_accuracy: 0.5517
    
    Epoch 00030: val_accuracy did not improve from 0.58621
    Epoch 31/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3555 - accuracy: 0.8706 - val_loss: 0.5829 - val_accuracy: 0.5172
    
    Epoch 00031: val_accuracy did not improve from 0.58621
    Epoch 32/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3428 - accuracy: 0.8588 - val_loss: 1.0976 - val_accuracy: 0.5603
    
    Epoch 00032: val_accuracy did not improve from 0.58621
    Epoch 33/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3476 - accuracy: 0.8659 - val_loss: 0.9030 - val_accuracy: 0.5431
    
    Epoch 00033: val_accuracy did not improve from 0.58621
    Epoch 34/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3301 - accuracy: 0.8753 - val_loss: 1.0762 - val_accuracy: 0.5690
    
    Epoch 00034: val_accuracy did not improve from 0.58621
    Epoch 35/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3678 - accuracy: 0.8447 - val_loss: 0.7499 - val_accuracy: 0.5259
    
    Epoch 00035: val_accuracy did not improve from 0.58621
    Epoch 36/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3656 - accuracy: 0.8306 - val_loss: 0.9960 - val_accuracy: 0.5603
    
    Epoch 00036: val_accuracy did not improve from 0.58621
    Epoch 37/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3436 - accuracy: 0.8541 - val_loss: 0.8240 - val_accuracy: 0.5517
    
    Epoch 00037: val_accuracy did not improve from 0.58621
    Epoch 38/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3320 - accuracy: 0.8612 - val_loss: 0.6757 - val_accuracy: 0.5431
    
    Epoch 00038: val_accuracy did not improve from 0.58621
    Epoch 39/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3328 - accuracy: 0.8565 - val_loss: 1.2394 - val_accuracy: 0.5517
    
    Epoch 00039: val_accuracy did not improve from 0.58621
    Epoch 40/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3844 - accuracy: 0.8212 - val_loss: 0.6667 - val_accuracy: 0.5517
    
    Epoch 00040: val_accuracy did not improve from 0.58621
    Epoch 41/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3832 - accuracy: 0.8235 - val_loss: 0.8128 - val_accuracy: 0.5690
    
    Epoch 00041: val_accuracy did not improve from 0.58621
    Epoch 42/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3435 - accuracy: 0.8518 - val_loss: 0.7913 - val_accuracy: 0.5345
    
    Epoch 00042: val_accuracy did not improve from 0.58621
    Epoch 43/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3397 - accuracy: 0.8612 - val_loss: 0.7908 - val_accuracy: 0.5345
    
    Epoch 00043: val_accuracy did not improve from 0.58621
    Epoch 44/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3234 - accuracy: 0.8800 - val_loss: 0.5274 - val_accuracy: 0.5603
    
    Epoch 00044: val_accuracy did not improve from 0.58621
    Epoch 45/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3426 - accuracy: 0.8494 - val_loss: 1.0416 - val_accuracy: 0.5345
    
    Epoch 00045: val_accuracy did not improve from 0.58621
    Epoch 46/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2873 - accuracy: 0.9035 - val_loss: 0.7047 - val_accuracy: 0.5345
    
    Epoch 00046: val_accuracy did not improve from 0.58621
    Epoch 47/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3083 - accuracy: 0.8706 - val_loss: 1.6751 - val_accuracy: 0.5603
    
    Epoch 00047: val_accuracy did not improve from 0.58621
    Epoch 48/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3518 - accuracy: 0.8729 - val_loss: 0.9985 - val_accuracy: 0.5690
    
    Epoch 00048: val_accuracy did not improve from 0.58621
    Epoch 49/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3198 - accuracy: 0.8706 - val_loss: 0.3768 - val_accuracy: 0.5259
    
    Epoch 00049: val_accuracy did not improve from 0.58621
    Epoch 50/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3013 - accuracy: 0.8941 - val_loss: 1.0320 - val_accuracy: 0.5517
    
    Epoch 00050: val_accuracy did not improve from 0.58621
    Epoch 51/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3309 - accuracy: 0.8659 - val_loss: 1.0530 - val_accuracy: 0.5259
    
    Epoch 00051: val_accuracy did not improve from 0.58621
    Epoch 52/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2949 - accuracy: 0.8824 - val_loss: 1.1942 - val_accuracy: 0.5345
    
    Epoch 00052: val_accuracy did not improve from 0.58621
    Epoch 53/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3122 - accuracy: 0.8800 - val_loss: 1.2658 - val_accuracy: 0.5603
    
    Epoch 00053: val_accuracy did not improve from 0.58621
    Epoch 54/100
    5/5 [==============================] - 9s 2s/step - loss: 0.2735 - accuracy: 0.8824 - val_loss: 1.1796 - val_accuracy: 0.5345
    
    Epoch 00054: val_accuracy did not improve from 0.58621
    Epoch 55/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2871 - accuracy: 0.8988 - val_loss: 0.9813 - val_accuracy: 0.5345
    
    Epoch 00055: val_accuracy did not improve from 0.58621
    Epoch 56/100
    5/5 [==============================] - 9s 2s/step - loss: 0.2890 - accuracy: 0.9059 - val_loss: 1.0376 - val_accuracy: 0.5603
    
    Epoch 00056: val_accuracy did not improve from 0.58621
    Epoch 57/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3070 - accuracy: 0.8941 - val_loss: 1.2580 - val_accuracy: 0.5431
    
    Epoch 00057: val_accuracy did not improve from 0.58621
    Epoch 58/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2935 - accuracy: 0.8941 - val_loss: 1.3105 - val_accuracy: 0.5431
    
    Epoch 00058: val_accuracy did not improve from 0.58621
    Epoch 59/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2767 - accuracy: 0.8871 - val_loss: 1.1539 - val_accuracy: 0.5259
    
    Epoch 00059: val_accuracy did not improve from 0.58621
    Epoch 60/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3062 - accuracy: 0.8847 - val_loss: 0.8769 - val_accuracy: 0.5431
    
    Epoch 00060: val_accuracy did not improve from 0.58621
    Epoch 61/100
    5/5 [==============================] - 9s 2s/step - loss: 0.2947 - accuracy: 0.8800 - val_loss: 0.8961 - val_accuracy: 0.5603
    
    Epoch 00061: val_accuracy did not improve from 0.58621
    Epoch 62/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2611 - accuracy: 0.8918 - val_loss: 1.4200 - val_accuracy: 0.5259
    
    Epoch 00062: val_accuracy did not improve from 0.58621
    Epoch 63/100
    5/5 [==============================] - 9s 2s/step - loss: 0.3037 - accuracy: 0.8965 - val_loss: 0.4794 - val_accuracy: 0.5776
    
    Epoch 00063: val_accuracy did not improve from 0.58621
    Epoch 64/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2694 - accuracy: 0.9059 - val_loss: 0.9954 - val_accuracy: 0.5603
    
    Epoch 00064: val_accuracy did not improve from 0.58621
    Epoch 65/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3479 - accuracy: 0.8518 - val_loss: 0.7984 - val_accuracy: 0.5603
    
    Epoch 00065: val_accuracy did not improve from 0.58621
    Epoch 66/100
    5/5 [==============================] - 9s 2s/step - loss: 0.2904 - accuracy: 0.8800 - val_loss: 1.4988 - val_accuracy: 0.5603
    
    Epoch 00066: val_accuracy did not improve from 0.58621
    Epoch 67/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3006 - accuracy: 0.8871 - val_loss: 1.6193 - val_accuracy: 0.5603
    
    Epoch 00067: val_accuracy did not improve from 0.58621
    Epoch 68/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2798 - accuracy: 0.8894 - val_loss: 1.3407 - val_accuracy: 0.5690
    
    Epoch 00068: val_accuracy did not improve from 0.58621
    Epoch 69/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2892 - accuracy: 0.9035 - val_loss: 1.7635 - val_accuracy: 0.5517
    
    Epoch 00069: val_accuracy did not improve from 0.58621
    Epoch 70/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2946 - accuracy: 0.8847 - val_loss: 1.6387 - val_accuracy: 0.5603
    
    Epoch 00070: val_accuracy did not improve from 0.58621
    Epoch 71/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2486 - accuracy: 0.9106 - val_loss: 1.2734 - val_accuracy: 0.5603
    
    Epoch 00071: val_accuracy did not improve from 0.58621
    Epoch 72/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2644 - accuracy: 0.9153 - val_loss: 0.4697 - val_accuracy: 0.5690
    
    Epoch 00072: val_accuracy did not improve from 0.58621
    Epoch 73/100
    5/5 [==============================] - 9s 2s/step - loss: 0.2885 - accuracy: 0.8941 - val_loss: 1.4267 - val_accuracy: 0.5345
    
    Epoch 00073: val_accuracy did not improve from 0.58621
    Epoch 74/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2600 - accuracy: 0.8800 - val_loss: 1.1460 - val_accuracy: 0.5603
    
    Epoch 00074: val_accuracy did not improve from 0.58621
    Epoch 75/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2777 - accuracy: 0.8847 - val_loss: 0.5679 - val_accuracy: 0.5690
    
    Epoch 00075: val_accuracy did not improve from 0.58621
    Epoch 76/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2703 - accuracy: 0.8918 - val_loss: 1.2378 - val_accuracy: 0.5345
    
    Epoch 00076: val_accuracy did not improve from 0.58621
    Epoch 77/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2750 - accuracy: 0.9012 - val_loss: 1.7775 - val_accuracy: 0.5690
    
    Epoch 00077: val_accuracy did not improve from 0.58621
    Epoch 78/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2581 - accuracy: 0.8988 - val_loss: 0.6691 - val_accuracy: 0.5776
    
    Epoch 00078: val_accuracy did not improve from 0.58621
    Epoch 79/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3209 - accuracy: 0.8729 - val_loss: 1.9737 - val_accuracy: 0.5345
    
    Epoch 00079: val_accuracy did not improve from 0.58621
    Epoch 80/100
    5/5 [==============================] - 9s 2s/step - loss: 0.2710 - accuracy: 0.8941 - val_loss: 0.8670 - val_accuracy: 0.5776
    
    Epoch 00080: val_accuracy did not improve from 0.58621
    Epoch 81/100
    5/5 [==============================] - 10s 2s/step - loss: 0.3306 - accuracy: 0.8612 - val_loss: 0.7545 - val_accuracy: 0.5603
    
    Epoch 00081: val_accuracy did not improve from 0.58621
    Epoch 82/100
    5/5 [==============================] - 9s 2s/step - loss: 0.2881 - accuracy: 0.8918 - val_loss: 1.5823 - val_accuracy: 0.5345
    
    Epoch 00082: val_accuracy did not improve from 0.58621
    Epoch 83/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2746 - accuracy: 0.8941 - val_loss: 1.1938 - val_accuracy: 0.5603
    
    Epoch 00083: val_accuracy did not improve from 0.58621
    Epoch 84/100
    5/5 [==============================] - 9s 2s/step - loss: 0.2635 - accuracy: 0.9176 - val_loss: 1.2982 - val_accuracy: 0.5690
    
    Epoch 00084: val_accuracy did not improve from 0.58621
    Epoch 85/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2825 - accuracy: 0.8918 - val_loss: 1.2733 - val_accuracy: 0.5345
    
    Epoch 00085: val_accuracy did not improve from 0.58621
    Epoch 86/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2725 - accuracy: 0.8965 - val_loss: 0.9127 - val_accuracy: 0.5776
    
    Epoch 00086: val_accuracy did not improve from 0.58621
    Epoch 87/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2655 - accuracy: 0.8941 - val_loss: 0.6599 - val_accuracy: 0.5259
    
    Epoch 00087: val_accuracy did not improve from 0.58621
    Epoch 88/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2768 - accuracy: 0.8941 - val_loss: 0.8989 - val_accuracy: 0.5690
    
    Epoch 00088: val_accuracy did not improve from 0.58621
    Epoch 89/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2506 - accuracy: 0.8965 - val_loss: 0.6099 - val_accuracy: 0.5603
    
    Epoch 00089: val_accuracy did not improve from 0.58621
    Epoch 90/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2606 - accuracy: 0.9106 - val_loss: 1.0991 - val_accuracy: 0.5259
    
    Epoch 00090: val_accuracy did not improve from 0.58621
    Epoch 91/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2662 - accuracy: 0.9035 - val_loss: 1.4280 - val_accuracy: 0.5603
    
    Epoch 00091: val_accuracy did not improve from 0.58621
    Epoch 92/100
    5/5 [==============================] - 9s 2s/step - loss: 0.2821 - accuracy: 0.8941 - val_loss: 1.0824 - val_accuracy: 0.5776
    
    Epoch 00092: val_accuracy did not improve from 0.58621
    Epoch 93/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2489 - accuracy: 0.9200 - val_loss: 1.1762 - val_accuracy: 0.5345
    
    Epoch 00093: val_accuracy did not improve from 0.58621
    Epoch 94/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2801 - accuracy: 0.8753 - val_loss: 0.7523 - val_accuracy: 0.5776
    
    Epoch 00094: val_accuracy did not improve from 0.58621
    Epoch 95/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2816 - accuracy: 0.9082 - val_loss: 0.7695 - val_accuracy: 0.5517
    
    Epoch 00095: val_accuracy did not improve from 0.58621
    Epoch 96/100
    5/5 [==============================] - 9s 2s/step - loss: 0.2588 - accuracy: 0.8988 - val_loss: 1.6505 - val_accuracy: 0.5345
    
    Epoch 00096: val_accuracy did not improve from 0.58621
    Epoch 97/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2698 - accuracy: 0.8894 - val_loss: 0.8359 - val_accuracy: 0.5603
    
    Epoch 00097: val_accuracy did not improve from 0.58621
    Epoch 98/100
    5/5 [==============================] - 9s 2s/step - loss: 0.2637 - accuracy: 0.8918 - val_loss: 1.0211 - val_accuracy: 0.5345
    
    Epoch 00098: val_accuracy did not improve from 0.58621
    Epoch 99/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2347 - accuracy: 0.9129 - val_loss: 1.5286 - val_accuracy: 0.5690
    
    Epoch 00099: val_accuracy did not improve from 0.58621
    Epoch 100/100
    5/5 [==============================] - 10s 2s/step - loss: 0.2418 - accuracy: 0.9176 - val_loss: 0.7346 - val_accuracy: 0.5431
    
    Epoch 00100: val_accuracy did not improve from 0.58621



```python
plot_training_loss_and_accuracy(inceptionv3_fit_history, 'Evolution of loss and accuracy with the number of training epochs for InceptionV3', 'inceptionv3_loss_acc.png')
```


![png]({{ site.baseurl }}/images/covid19_udacity/output_45_0.png)



```python
def create_validation_accuracy_during_training(vgg16_fit_history, resnet50_fit_history, densenet121_fit_history, nasnetmobile_fit_history, inceptionv3_fit_history):
  '''
  INPUT:
  vgg16_fit_history - training statistics dictionary custom model VGG16
  resnet50_fit_history - training statistics dictionary custom model ResNet50
  densenet121_fit_history - training statistics dictionary custom model DenseNet121
  nasnetmobile_fit_history - training statistics dictionary custom model NASNetMobile
  inceptionv3_fit_history - training statistics dictionary custom model InceptionV3

  OUTPUT:
  df_val_accuracy - dataframe containing training 'val_loss', 'val_accuracy', 'loss', 'accuracy' for each of the models

  This function gather the training 'val_loss', 'val_accuracy', 'loss', 'accuracy' for each of the models and put them
  into a dataframe
  '''

  def compute_dataframe_entry(fit_history):
    '''
    INPUT:
    fit_history - training statistics dictionary custom model

    OUTPUT:
    a row for the dataframe containing training 'val_loss', 'val_accuracy', 'loss', 'accuracy' for one model
    
    This function returns a row for the dataframe containing training 'val_loss', 'val_accuracy', 'loss', 'accuracy' for one model
    '''
    # create a dataframe from fit_history.history
    df_temp = pd.DataFrame.from_dict(fit_history.history)
    #df_temp.head()

    # find the maximum validation accuracy
    max_val_accuracy = df_temp.val_accuracy.max()

    # find the first row from df_temp where the val_accuracy is the max. value
    # assign to a temp dict.
    temp_dict=df_temp.loc[df_temp[df_temp.val_accuracy==max_val_accuracy].index].iloc[0]
    #temp_dict

    return temp_dict.get('val_loss'), max_val_accuracy, temp_dict.get('loss'), temp_dict.get('accuracy')

  #create an empty dataframe
  df_val_accuracy = pd.DataFrame(columns=['model', 'val_loss', 'val_accuracy', 'loss', 'accuracy'])
  #df_val_accuracy

  val_loss, val_accuracy, loss, accuracy = compute_dataframe_entry(vgg16_fit_history)

  df_val_accuracy = df_val_accuracy.append(
      {'model': 'VGG16', 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'loss': loss, 'accuracy': accuracy}, 
      ignore_index=True)

  val_loss, val_accuracy, loss, accuracy = compute_dataframe_entry(resnet50_fit_history)

  df_val_accuracy = df_val_accuracy.append(
      {'model': 'ResNet50', 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'loss': loss, 'accuracy': accuracy}, 
      ignore_index=True)

  val_loss, val_accuracy, loss, accuracy = compute_dataframe_entry(densenet121_fit_history)

  df_val_accuracy = df_val_accuracy.append(
      {'model': 'DenseNet121', 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'loss': loss, 'accuracy': accuracy}, 
      ignore_index=True)

  val_loss, val_accuracy, loss, accuracy = compute_dataframe_entry(nasnetmobile_fit_history)

  df_val_accuracy = df_val_accuracy.append(
      {'model': 'NASNetMobile', 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'loss': loss, 'accuracy': accuracy}, 
      ignore_index=True)

  val_loss, val_accuracy, loss, accuracy = compute_dataframe_entry(nasnetmobile_fit_history)

  df_val_accuracy = df_val_accuracy.append(
      {'model': 'NASNetMobile', 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'loss': loss, 'accuracy': accuracy}, 
      ignore_index=True)

  val_loss, val_accuracy, loss, accuracy = compute_dataframe_entry(inceptionv3_fit_history)

  df_val_accuracy = df_val_accuracy.append(
      {'model': 'InceptionV3', 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'loss': loss, 'accuracy': accuracy}, 
      ignore_index=True)

  return df_val_accuracy

 
```


```python
df_val_accuracy = create_validation_accuracy_during_training(vgg16_fit_history, resnet50_fit_history, densenet121_fit_history, nasnetmobile_fit_history, inceptionv3_fit_history)
print('Statistics of Training:')
display(df_val_accuracy)
```

    Statistics of Training:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>val_loss</th>
      <th>val_accuracy</th>
      <th>loss</th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VGG16</td>
      <td>0.851205</td>
      <td>0.715517</td>
      <td>0.273640</td>
      <td>0.884706</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ResNet50</td>
      <td>1.254023</td>
      <td>0.672414</td>
      <td>0.117272</td>
      <td>0.964706</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DenseNet121</td>
      <td>0.613608</td>
      <td>0.655172</td>
      <td>0.239100</td>
      <td>0.898823</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NASNetMobile</td>
      <td>0.767678</td>
      <td>0.646552</td>
      <td>0.400589</td>
      <td>0.804706</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NASNetMobile</td>
      <td>0.767678</td>
      <td>0.646552</td>
      <td>0.400589</td>
      <td>0.804706</td>
    </tr>
    <tr>
      <th>5</th>
      <td>InceptionV3</td>
      <td>0.590555</td>
      <td>0.586207</td>
      <td>0.327752</td>
      <td>0.887059</td>
    </tr>
  </tbody>
</table>
</div>



```python
# save the training stats to csv (ignore index)
df_val_accuracy.to_csv('train_stats.csv', index=False)
```

Now that the models are trained, we are ready to start using it to classify images.


## Evaluation

#### Accuracies of our Models

In this part, we will evaluate our deep learning models on a test data. For this part, we will need to do the following:

1. Load our saved model that was built using the five models. 
2. Construct an ImageDataGenerator for the test set. For this ImageDataGenerator instance, you only need to pass the directory of the test images, target size, and the **shuffle** parameter and set it to False.
3. Use the **evaluate_generator** method to evaluate your models on the test data, by passing the above ImageDataGenerator as an argument. You can learn more about **evaluate_generator** [here](https://keras.io/models/sequential/).
4. Print the performances of our five classifiers.



```python
# load the saved model
model_vgg16 = load_model('classifier_vgg16_model.h5')
model_resnet50 = load_model('classifier_resnet50_model.h5')
model_densenet121 = load_model('classifier_densenet121_model.h5')
model_nasnetmobile = load_model('classifier_nasnetmobile_model.h5')
model_inceptionv3 = load_model('classifier_inceptionv3_model.h5')

```

    WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    


    /usr/local/lib/python3.6/dist-packages/keras/engine/saving.py:384: UserWarning: Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
      warnings.warn('Error in loading the saved optimizer '
    /usr/local/lib/python3.6/dist-packages/keras/engine/saving.py:384: UserWarning: Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
      warnings.warn('Error in loading the saved optimizer '


    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4074: The name tf.nn.avg_pool is deprecated. Please use tf.nn.avg_pool2d instead.
    


    /usr/local/lib/python3.6/dist-packages/keras/engine/saving.py:384: UserWarning: Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
      warnings.warn('Error in loading the saved optimizer '
    /usr/local/lib/python3.6/dist-packages/keras/engine/saving.py:384: UserWarning: Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
      warnings.warn('Error in loading the saved optimizer '
    /usr/local/lib/python3.6/dist-packages/keras/engine/saving.py:384: UserWarning: Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
      warnings.warn('Error in loading the saved optimizer '


Use the flow_from_directory method to get the test images and assign the result to test_generator.


```python
# define the image size for VGG16, ResNet50, DenseNet121 and NASNetMobile
image_resize = 224
```


```python
# VGG16 test generator
vgg16_data_generator = ImageDataGenerator(
    preprocessing_function=vgg16_preprocess_input
    )

vgg16_test_generator = vgg16_data_generator.flow_from_directory(
    './data_processed/test',
    target_size=(image_resize, image_resize),
    shuffle=False)
```

    Found 200 images belonging to 2 classes.



```python
# ResNet50 test generator
resnet50_data_generator = ImageDataGenerator(
    preprocessing_function=resnet50_preprocess_input
    )

resnet50_test_generator = resnet50_data_generator.flow_from_directory(
    './data_processed/test',
    target_size=(image_resize, image_resize),
    shuffle=False)
```

    Found 200 images belonging to 2 classes.



```python
# DenseNet121 test generator
densenet121_data_generator = ImageDataGenerator(
    preprocessing_function=densenet121_preprocess_input
    )

densenet121_test_generator = densenet121_data_generator.flow_from_directory(
    './data_processed/test',
    target_size=(image_resize, image_resize),
    shuffle=False)
```

    Found 200 images belonging to 2 classes.



```python
# NASNetMobile test generator
nasnetmobile_data_generator = ImageDataGenerator(
    preprocessing_function=nasnetmobile_preprocess_input
    )

nasnetmobile_test_generator = nasnetmobile_data_generator.flow_from_directory(
    './data_processed/test',
    target_size=(image_resize, image_resize),
    shuffle=False)
```

    Found 200 images belonging to 2 classes.



```python
# InceptionV3 test generator, note that image size is 299x299
inceptionv3_data_generator = ImageDataGenerator(
    preprocessing_function=inceptionv3_preprocess_input
    )

inceptionv3_test_generator = inceptionv3_data_generator.flow_from_directory(
    './data_processed/test',
    target_size=(299, 299),
    shuffle=False)
```

    Found 200 images belonging to 2 classes.


Use the evaluate_generator method to evaluate our models on the test data, by passing the above ImageDataGenerator as an argument.


```python
vgg16_evaulation = model_vgg16.evaluate_generator(vgg16_test_generator, len(vgg16_test_generator), verbose=1)
print("Accuracy of vgg16 on the test set = ", vgg16_evaulation[1])
```

    7/7 [==============================] - 10s 1s/step
    Accuracy of vgg16 on the test set =  0.6949999928474426



```python
resnet50_evaulation = model_resnet50.evaluate_generator(resnet50_test_generator, len(resnet50_test_generator), verbose=1)
print("Accuracy of resnet50 on the test set = ", resnet50_evaulation[1])
```

    7/7 [==============================] - 4s 568ms/step
    Accuracy of resnet50 on the test set =  0.7400000095367432



```python
densenet121_evaulation = model_densenet121.evaluate_generator(densenet121_test_generator, len(densenet121_test_generator), verbose=1)
print("Accuracy of densenet121 on the test set = ", densenet121_evaulation[1])
```

    7/7 [==============================] - 9s 1s/step
    Accuracy of densenet121 on the test set =  0.7149999737739563



```python
nasnetmobile_evaulation = model_nasnetmobile.evaluate_generator(nasnetmobile_test_generator, len(nasnetmobile_test_generator), verbose=1)
print("Accuracy of nasnetmobile on the test set = ", nasnetmobile_evaulation[1])
```

    7/7 [==============================] - 9s 1s/step
    Accuracy of nasnetmobile on the test set =  0.675000011920929



```python
inceptionv3_evaulation = model_inceptionv3.evaluate_generator(inceptionv3_test_generator, len(inceptionv3_test_generator), verbose=1)
print("Accuracy of inceptionv3 on the test set = ", inceptionv3_evaulation[1])
```

    7/7 [==============================] - 9s 1s/step
    Accuracy of inceptionv3 on the test set =  0.75


#### Classifications of Test Dataset

In this section, we will predict whether the images in the test data are images of COVID CT scan or not. We will do the following:

  * Use the **predict_generator** method to predict the class of the images in the test data, by passing the test data ImageDataGenerator instance defined in the previous part as an argument. You can learn more about the **predict_generator** method [here](https://keras.io/models/sequential/).


```python
vgg16_prediction = model_vgg16.predict_generator(vgg16_test_generator, len(vgg16_test_generator), verbose=1)
```

    7/7 [==============================] - 7s 1s/step



```python
# reverse the key-value pairs from train_generator.class_indices
class_label_dict = {v: k for k, v in vgg16_test_generator.class_indices.items()}
class_label_dict
```




    {0: 'COVID', 1: 'NonCOVID'}




```python
# print the prediction of the first 5 samples using the VGG16 model
for index in range(5):
  predicted_class_index=np.argmax(vgg16_prediction[index])

  if predicted_class_index in [0, 1]:
    print('image {} is predicted as {} with probability {}'.format(vgg16_test_generator.filenames[index], class_label_dict.get(predicted_class_index), vgg16_prediction[index][predicted_class_index]))
  else:
    print('image {} cannot be classified.'.format(test_generator.filenames[index]))
  
```

    image COVID/2019-novel-Coronavirus-severe-adult-respiratory-dist_2020_International-Jour-p3-89%0.png is predicted as COVID with probability 0.9408342242240906
    image COVID/2019-novel-Coronavirus-severe-adult-respiratory-dist_2020_International-Jour-p3-89%1.png is predicted as NonCOVID with probability 0.6286885738372803
    image COVID/2019-novel-Coronavirus-severe-adult-respiratory-dist_2020_International-Jour-p3-91.png is predicted as COVID with probability 0.9957075119018555
    image COVID/2020.03.22.20040782-p25-1542.png is predicted as COVID with probability 0.9816886186599731
    image COVID/2020.03.22.20040782-p25-1543.png is predicted as NonCOVID with probability 0.9767597317695618



```python
resnet50_prediction = model_resnet50.predict_generator(resnet50_test_generator, len(resnet50_test_generator), verbose=1)
```

    7/7 [==============================] - 8s 1s/step



```python
densenet121_prediction = model_densenet121.predict_generator(densenet121_test_generator, len(densenet121_test_generator), verbose=1)
```

    7/7 [==============================] - 9s 1s/step



```python
nasnetmobile_prediction = model_nasnetmobile.predict_generator(nasnetmobile_test_generator, len(nasnetmobile_test_generator), verbose=1)
```

    7/7 [==============================] - 11s 2s/step



```python
inceptionv3_prediction = model_inceptionv3.predict_generator(inceptionv3_test_generator, len(inceptionv3_test_generator), verbose=1)
```

    7/7 [==============================] - 9s 1s/step


Display the classification report and Confusion matrix


```python
def display_result(Y_pred, test_generator, plot_filename):
  '''
  INPUT:
  Y_pred - class prediction numpy matrix
  test_generator - test data generator
  plot_filename - filepath to store the confusion matrix plot

  OUTPUT:
  None
  
  This function prints the classification report and confusion matrix
  '''
  def display_confusion_matrix(y_test, y_pred, class_label_dict, plot_filename):
      '''
      INPUT:
      y_test - ground truth numpy array
      y_pred - class prediction numpy array
      class_label_dict - dictionary for mapping class to label
      plot_filename - filepath to store the confusion matrix plot
    
      OUTPUT:
      None
      This function display confusion matrix and save it
      '''

      confusion_mat = confusion_matrix(y_test, y_pred)
      accuracy = (y_pred == y_test).mean()

      #print("Labels:", labels)
      #print("Confusion Matrix:\n", confusion_mat)
      print("Accuracy:", accuracy)
    
      ax = sns.heatmap(confusion_mat, cmap="YlGn", annot=True, fmt="d")
      ax.set_xticklabels(class_label_dict.values())
      ax.set_yticklabels(class_label_dict.values(), rotation=0)

      ax.figure.show()
      ax.figure.savefig(plot_filename, dpi=200, format='png', bbox_inches='tight')

  # reverse the key-value pairs from train_generator.class_indices
  class_label_dict = {v: k for k, v in test_generator.class_indices.items()}

  # Get the predicted class
  y_pred = np.argmax(Y_pred, axis=1)

  # Display confusion matrix
  display_confusion_matrix(test_generator.classes, y_pred, class_label_dict, plot_filename)

  print('Classification Report')
  print(classification_report(test_generator.classes, y_pred, target_names=class_label_dict.values()))

  # get the classification_report as a dictionary
  report_dict = classification_report(test_generator.classes, y_pred, target_names=class_label_dict.values(), output_dict=True)

  # convert report_dict to dataframe
  df = pd.DataFrame(report_dict)

  # drop columns 'macro avg' and 'weighted avg'
  df = df.drop(labels=['macro avg', 'weighted avg'], axis=1)

  return df

```


```python
# VGG16 result
report_vgg16 = display_result(vgg16_prediction, vgg16_test_generator, 'vgg16_confusion_matrix.png')
```

    Accuracy: 0.695
    Classification Report
                  precision    recall  f1-score   support
    
           COVID       0.67      0.72      0.69        95
        NonCOVID       0.72      0.68      0.70       105
    
        accuracy                           0.69       200
       macro avg       0.70      0.70      0.69       200
    weighted avg       0.70      0.69      0.70       200
    



![png]({{ site.baseurl }}/images/covid19_udacity/output_78_1.png)



```python
# ResNet50 result
report_resnet50 = display_result(resnet50_prediction, resnet50_test_generator, 'resnet50_confusion_matrix.png')
```

    Accuracy: 0.74
    Classification Report
                  precision    recall  f1-score   support
    
           COVID       0.72      0.74      0.73        95
        NonCOVID       0.76      0.74      0.75       105
    
        accuracy                           0.74       200
       macro avg       0.74      0.74      0.74       200
    weighted avg       0.74      0.74      0.74       200
    



![png]({{ site.baseurl }}/images/covid19_udacity/output_79_1.png)



```python
# DenseNet121 result
report_densenet121 = display_result(densenet121_prediction, densenet121_test_generator, 'densenet121_confusion_matrix.png')
```

    Accuracy: 0.715
    Classification Report
                  precision    recall  f1-score   support
    
           COVID       0.70      0.69      0.70        95
        NonCOVID       0.73      0.73      0.73       105
    
        accuracy                           0.71       200
       macro avg       0.71      0.71      0.71       200
    weighted avg       0.71      0.71      0.71       200
    



![png]({{ site.baseurl }}/images/covid19_udacity/output_80_1.png)



```python
# NASNetMobile results
report_nasnetmobile = display_result(nasnetmobile_prediction, nasnetmobile_test_generator, 'nasnetmobile_confusion_matrix.png')
```

    Accuracy: 0.675
    Classification Report
                  precision    recall  f1-score   support
    
           COVID       0.69      0.57      0.62        95
        NonCOVID       0.66      0.77      0.71       105
    
        accuracy                           0.68       200
       macro avg       0.68      0.67      0.67       200
    weighted avg       0.68      0.68      0.67       200
    



![png]({{ site.baseurl }}/images/covid19_udacity/output_81_1.png)



```python
# InceptionV3 results
report_inceptionv3 = display_result(inceptionv3_prediction, inceptionv3_test_generator, 'inceptionv3_confusion_matrix.png')
```

    Accuracy: 0.75
    Classification Report
                  precision    recall  f1-score   support
    
           COVID       0.72      0.77      0.74        95
        NonCOVID       0.78      0.73      0.75       105
    
        accuracy                           0.75       200
       macro avg       0.75      0.75      0.75       200
    weighted avg       0.75      0.75      0.75       200
    



![png]({{ site.baseurl }}/images/covid19_udacity/output_82_1.png)


##### Majority voting from a committee of our Models


```python
# Create a dataframe that includes all the classification results from the 5 models
def create_majority_vote_dataframe(vgg16_prediction, resnet50_prediction, densenet121_prediction, nasnetmobile_prediction, inceptionv3_prediction):
  '''
  INPUT:
  vgg16_prediction - VGG16 class prediction numpy matrix
  resnet50_prediction - ResNet50 class prediction numpy matrix 
  densenet121_prediction - DenseNet121 class prediction numpy matrix 
  nasnetmobile_prediction - NASNetMobile class prediction numpy matrix 
  inceptionv3_prediction - InceptionV3 class prediction numpy matrix


  OUTPUT:
  df_pred - dataframe containing all the classification results from the 5 models
    
  This function combines the classficiation results from 5 models into one dataframe
  '''
  df1 = pd.DataFrame(data=vgg16_prediction, columns=["vgg16_COVID", "vgg16_NonCOVID"])
  df1['vgg16_pred'] = np.argmax(vgg16_prediction, axis=1)
  #df1.head()

  df2 = pd.DataFrame(data=resnet50_prediction, columns=["resnet50_COVID", "resnet50_NonCOVID"])
  df2['resnet50_pred'] = np.argmax(resnet50_prediction, axis=1)
  #df2.head()

  df3 = pd.DataFrame(data=densenet121_prediction, columns=["densenet121_COVID", "densenet121_NonCOVID"])
  df3['densenet121_pred'] = np.argmax(densenet121_prediction, axis=1)
  #df3.head()

  df4 = pd.DataFrame(data=nasnetmobile_prediction, columns=["nasnetmobile_COVID", "nasnetmobile_NonCOVID"])
  df4['nasnetmobile_pred'] = np.argmax(nasnetmobile_prediction, axis=1)
  #df4.head()

  df5 = pd.DataFrame(data=inceptionv3_prediction, columns=["inceptionv3_COVID", "inceptionv3_NonCOVID"])
  df5['inceptionv3_pred'] = np.argmax(inceptionv3_prediction, axis=1)
  #df5.head()  

  df_pred = pd.concat([df1, df2, df3, df4, df5], axis=1)
  #df_pred.head()

  df_pred['majority_vote'] = df_pred[['vgg16_pred', 'resnet50_pred', 'densenet121_pred', 'nasnetmobile_pred', 'inceptionv3_pred']].mode(axis='columns', numeric_only=True)[0]
  #df_pred.head()

  #df_pred['majority_vote'].unique()

  COVID_columns = ['vgg16_COVID', 'resnet50_COVID', 'densenet121_COVID', 'nasnetmobile_COVID', 'inceptionv3_COVID']
  NonCOVID_columns = ['vgg16_NonCOVID', 'resnet50_COVID', 'densenet121_NonCOVID', 'nasnetmobile_NonCOVID', 'inceptionv3_NonCOVID']

  df_pred['combined_mode_COVID'] = df_pred.apply(lambda row: row[COVID_columns].max() if row.majority_vote== 0 else 0, axis=1)
  df_pred['combined_mode_NonCOVID'] = df_pred.apply(lambda row: row[NonCOVID_columns].max() if row.majority_vote== 1 else 0, axis=1)
  #df_pred.head()

  return df_pred
```


```python
# Create a dataframe that includes all the classification results from the 5 models
df_majority_vote = create_majority_vote_dataframe(vgg16_prediction, resnet50_prediction, densenet121_prediction, nasnetmobile_prediction, inceptionv3_prediction)

# Create a new column filename to store the filenames of the samples
df_majority_vote['filename'] = vgg16_test_generator.filenames

# save the test dataset classifications to csv (ignore index)
df_majority_vote.to_csv('test_dataset_classification_results.csv', index=False)
```


```python
# Get the predictions usig majority vote
majority_vote_prediction = df_majority_vote[['combined_mode_COVID', 'combined_mode_NonCOVID']].to_numpy()
report_majority_vote = display_result(majority_vote_prediction, vgg16_test_generator, 'combined_mode_confusion_matrix.png')
```

    Accuracy: 0.765
    Classification Report
                  precision    recall  f1-score   support
    
           COVID       0.74      0.77      0.76        95
        NonCOVID       0.78      0.76      0.77       105
    
        accuracy                           0.77       200
       macro avg       0.76      0.77      0.76       200
    weighted avg       0.77      0.77      0.77       200
    



![png]({{ site.baseurl }}/images/covid19_udacity/output_86_1.png)



```python
def create_test_stats(report_vgg16, report_resnet50, report_densenet121, report_nasnetmobile, report_inceptionv3, report_majority_vote):
  '''
  INPUT:
  report_vgg16 - VGG16 classification report
  report_resnet50 - ResNet50 classification report 
  report_densenet121 - DenseNet121 classification report
  report_nasnetmobile - NASNetMobile classification report
  report_inceptionv3 - InceptionV3 classification report
  report_majority_vote - Majority Voting classification report


  OUTPUT:
  df_report - dataframe containing all the classification reports from the 5 models
    
  This function combines the classficiation reports from 6 models into one dataframe
  '''
  # add prefix vgg16_' to each column names to report_vgg16
  df1 = report_vgg16.add_prefix('vgg16_')

  # add prefix 'resnet50_' to each column names to report_resnet50
  df2 = report_resnet50.add_prefix('resnet50_')

  # add prefix 'densenet121_' to each column names to report_densenet121
  df3 = report_densenet121.add_prefix('densenet121_')

  # add prefix 'nasnetmobile_' to each column names to report_nasnetmobile
  df4 = report_nasnetmobile.add_prefix('nasnetmobile_')

  # add prefix 'inceptionv3_' to each column names to report_inceptionv3
  df5 = report_inceptionv3.add_prefix('inceptionv3_')

  # add prefix 'majority_vote_' to each column names to report_majority_vote
  df6 = report_majority_vote.add_prefix('majority_vote_')

  df_report = pd.concat([df1, df2, df3, df4, df5, df6], axis=1)
  
  return df_report

```


```python
df_report = create_test_stats(report_vgg16, report_resnet50, report_densenet121, report_nasnetmobile, report_inceptionv3, report_majority_vote)
display(df_report.T)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>vgg16_COVID</th>
      <td>0.666667</td>
      <td>0.715789</td>
      <td>0.690355</td>
      <td>95.000</td>
    </tr>
    <tr>
      <th>vgg16_NonCOVID</th>
      <td>0.724490</td>
      <td>0.676190</td>
      <td>0.699507</td>
      <td>105.000</td>
    </tr>
    <tr>
      <th>vgg16_accuracy</th>
      <td>0.695000</td>
      <td>0.695000</td>
      <td>0.695000</td>
      <td>0.695</td>
    </tr>
    <tr>
      <th>resnet50_COVID</th>
      <td>0.721649</td>
      <td>0.736842</td>
      <td>0.729167</td>
      <td>95.000</td>
    </tr>
    <tr>
      <th>resnet50_NonCOVID</th>
      <td>0.757282</td>
      <td>0.742857</td>
      <td>0.750000</td>
      <td>105.000</td>
    </tr>
    <tr>
      <th>resnet50_accuracy</th>
      <td>0.740000</td>
      <td>0.740000</td>
      <td>0.740000</td>
      <td>0.740</td>
    </tr>
    <tr>
      <th>densenet121_COVID</th>
      <td>0.702128</td>
      <td>0.694737</td>
      <td>0.698413</td>
      <td>95.000</td>
    </tr>
    <tr>
      <th>densenet121_NonCOVID</th>
      <td>0.726415</td>
      <td>0.733333</td>
      <td>0.729858</td>
      <td>105.000</td>
    </tr>
    <tr>
      <th>densenet121_accuracy</th>
      <td>0.715000</td>
      <td>0.715000</td>
      <td>0.715000</td>
      <td>0.715</td>
    </tr>
    <tr>
      <th>nasnetmobile_COVID</th>
      <td>0.692308</td>
      <td>0.568421</td>
      <td>0.624277</td>
      <td>95.000</td>
    </tr>
    <tr>
      <th>nasnetmobile_NonCOVID</th>
      <td>0.663934</td>
      <td>0.771429</td>
      <td>0.713656</td>
      <td>105.000</td>
    </tr>
    <tr>
      <th>nasnetmobile_accuracy</th>
      <td>0.675000</td>
      <td>0.675000</td>
      <td>0.675000</td>
      <td>0.675</td>
    </tr>
    <tr>
      <th>inceptionv3_COVID</th>
      <td>0.722772</td>
      <td>0.768421</td>
      <td>0.744898</td>
      <td>95.000</td>
    </tr>
    <tr>
      <th>inceptionv3_NonCOVID</th>
      <td>0.777778</td>
      <td>0.733333</td>
      <td>0.754902</td>
      <td>105.000</td>
    </tr>
    <tr>
      <th>inceptionv3_accuracy</th>
      <td>0.750000</td>
      <td>0.750000</td>
      <td>0.750000</td>
      <td>0.750</td>
    </tr>
    <tr>
      <th>majority_vote_COVID</th>
      <td>0.744898</td>
      <td>0.768421</td>
      <td>0.756477</td>
      <td>95.000</td>
    </tr>
    <tr>
      <th>majority_vote_NonCOVID</th>
      <td>0.784314</td>
      <td>0.761905</td>
      <td>0.772947</td>
      <td>105.000</td>
    </tr>
    <tr>
      <th>majority_vote_accuracy</th>
      <td>0.765000</td>
      <td>0.765000</td>
      <td>0.765000</td>
      <td>0.765</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_report.to_csv('test.stats.csv', index=False)
```

#### Classify One Image


```python
# select from COVID test dataset
img_dir = './data_processed/test/COVID/'
 
# randomly pick an image
img = random.choice(os.listdir(img_dir))
img_path = img_dir + img
 
with Image.open(img_path) as image1:
    plt.imshow(image1)
    print(image1.mode)
```

    RGB



![png]({{ site.baseurl }}/images/covid19_udacity/output_91_1.png)



```python
# function to classify one image using the 5 models and majority voting
def classify_one_image(img_path, class_label_dict, model_vgg16, model_resnet50, model_densenet121, model_nasnetmobile, model_inceptionv3):
  '''
  INPUT:
  img_path - filepath of the input image
  class_label_dict - dictionary for mapping class to label
  model_vgg16 - VGG16 model
  model_resnet50 - ResNet50 model
  model_densenet121 - DenseNet121 model
  model_nasnetmobile - NASNetMobile model
  model_inceptionv3 - InceptionV3 model

  OUTPUT:
  df_result - dataframe containing all the classification results from the 6 models
  voted_class - predicted class from the Majority of the comittee
  voted_prob - the highest prob. from the model that predicted voted_class
  voted_model - the model that gives the voted_prob
    
  This function combines the predicted class from each model and the voted class with the prob.
  '''
  img = image.load_img(img_path, target_size=(224, 224))
  x_224 = image.img_to_array(img)
  x_224 = np.expand_dims(x_224, axis=0)

  #create an empty dataframe
  df_result = pd.DataFrame(columns=['model', 'type', 'prob'])
  #df_result

  x_vgg16 = vgg16_preprocess_input(x_224)
  Y_pred = model_vgg16.predict(x_vgg16)
  #Y_pred

  y_pred=np.argmax(Y_pred)
  #print('image {} is predicted as {} with probability {}'.format(img_path, class_label_dict.get(y_pred), Y_pred[0][y_pred]))
  df_result = df_result.append({'model': 'VGG16', 'type' : class_label_dict.get(y_pred), 'prob' : Y_pred[0][y_pred]} , ignore_index=True)
  #df_result

  x_resnet50 = resnet50_preprocess_input(x_224)
  Y_pred = model_resnet50.predict(x_resnet50)
  #Y_pred

  y_pred=np.argmax(Y_pred)
  #print('image {} is predicted as {} with probability {}'.format(img_path, class_label_dict.get(y_pred), Y_pred[0][y_pred]))
  df_result = df_result.append({'model': 'ResNet50', 'type' : class_label_dict.get(y_pred), 'prob' : Y_pred[0][y_pred]} , ignore_index=True)
  #df_result

  x_densenet121 = densenet121_preprocess_input(x_224)
  Y_pred = model_densenet121.predict(x_densenet121)
  #Y_pred

  y_pred=np.argmax(Y_pred)
  #print('image {} is predicted as {} with probability {}'.format(img_path, class_label_dict.get(y_pred), Y_pred[0][y_pred]))
  df_result = df_result.append({'model': 'DenseNet121', 'type' : class_label_dict.get(y_pred), 'prob' : Y_pred[0][y_pred]} , ignore_index=True)
  #df_result

  x_nasnetmobile = nasnetmobile_preprocess_input(x_224)
  Y_pred = model_nasnetmobile.predict(x_nasnetmobile)
  #Y_pred

  y_pred=np.argmax(Y_pred)
  #print('image {} is predicted as {} with probability {}'.format(img_path, class_label_dict.get(y_pred), Y_pred[0][y_pred]))
  df_result = df_result.append({'model': 'NASNetMobile', 'type' : class_label_dict.get(y_pred), 'prob' : Y_pred[0][y_pred]} , ignore_index=True)
  #df_result

  img = image.load_img(img_path, target_size=(299, 299))
  x_299 = image.img_to_array(img)
  x_299 = np.expand_dims(x_299, axis=0)

  x_inceptionv3 = inceptionv3_preprocess_input(x_299)
  Y_pred = model_inceptionv3.predict(x_inceptionv3)
  #Y_pred

  y_pred=np.argmax(Y_pred)
  #print('image {} is predicted as {} with probability {}'.format(img_path, class_label_dict.get(y_pred), Y_pred[0][y_pred]))
  df_result = df_result.append({'model': 'InceptionV3', 'type' : class_label_dict.get(y_pred), 'prob' : Y_pred[0][y_pred]} , ignore_index=True)
  #df_result

  # find the class voted by most models
  voted_class = df_result.type.mode()[0]

  # find the highest prob. for the class voted
  voted_prob = df_result[df_result.type == voted_class].prob.max()

  # Get the first model from the series 
  voted_model = df_result.loc[df_result[df_result.prob == voted_prob].index]['model'].iloc[0]

  return df_result, voted_class, voted_prob, voted_model
```


```python
# get the dataframe of the prediction results from the 5 models 
# and the majority voted class, prob and the model that gives the highest prob.
df_result, voted_class, voted_prob, voted_model = classify_one_image(img_path, class_label_dict, model_vgg16, model_resnet50, model_densenet121, model_nasnetmobile, model_inceptionv3)

print('The classification results:')
print(df_result)
print('\nimage {} is voted as {} with probability {} from {} model'.format(img_path, voted_class, voted_prob, voted_model))


```

    The classification results:
              model   type      prob
    0         VGG16  COVID  0.999607
    1      ResNet50  COVID  0.999280
    2   DenseNet121  COVID  0.942965
    3  NASNetMobile  COVID  0.892145
    4   InceptionV3  COVID  0.722406
    
    image ./data_processed/test/COVID/2020.03.26.20041426-p11-125.png is voted as COVID with probability 0.9996071457862854 from VGG16 model



```python
# store the image filename into the dataframe
df_result['filename']=img_path
```


```python
df_result.to_csv('single_image_classification_results.csv', index=False)
```

## Conclusions

* Majority voting from a committee of our five models gives the highest precision, recall, f1-score for COVID and NonVID than the individual models.

* The f1-score for Majority voting for COVID prediction is 0.756477, NonCOVID 0.772947 and the accuracy of the predictions is 0.765000.

* We should be able fine-tune the individual five models to get a better performances of each model, which in turn, should also improve the performance of the Majority Voting committee.
