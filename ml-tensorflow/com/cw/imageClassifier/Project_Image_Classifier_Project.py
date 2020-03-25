#!/usr/bin/env python
# coding: utf-8

# # Your First AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from Oxford of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load the image dataset and create a pipeline.
# * Build and Train an image classifier on this dataset.
# * Use your trained model to perform inference on flower images.
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

# ## Import Resources

# In[1]:


# TODO: Make all necessary imports.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

warnings.filterwarnings('ignore')
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')


# ## Load the Dataset
# 
# Here you'll use `tensorflow_datasets` to load the [Oxford Flowers 102 dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102). This dataset has 3 splits: `'train'`, `'test'`, and `'validation'`.  You'll also need to make sure the training data is normalized and resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet, but you'll still need to normalize and resize the images to the appropriate size.

# In[2]:


# TODO: Load the dataset with TensorFlow Datasets.
dataset,datainfo = tfds.load('oxford_flowers102', as_supervised = True, with_info = True)

# TODO: Create a training set, a validation set and a test set.
training_set, test_set, validation_set = dataset['train'], dataset['test'], dataset['validation']


# In[3]:


# Check that dataset is a dictionary
print('dataset has type:', type(dataset))

# Print the keys of the dataset dictionary
print('\nThe keys of dataset are:', list(dataset.keys()))

print('training data set has type:', type(training_set))


# ## Explore the Dataset

# In[4]:


datainfo


# In[5]:


# TODO: Get the number of examples in each set from the dataset info.
num_training_examples  = datainfo.splits['train'].num_examples
num_test_examples  = datainfo.splits['test'].num_examples
num_validation_examples  = datainfo.splits['validation'].num_examples

print('\nThere are {:,} images in the training set'.format(num_training_examples))
print('There are {:,} images in the test set'.format(num_test_examples))
print('There are {:,} images in the validation set'.format(num_validation_examples))

# TODO: Get the number of classes in the dataset from the dataset info.
num_classes = datainfo.features['label'].num_classes
print('\nThere are {:,} classes in our dataset'.format(num_classes))


# In[6]:


# TODO: Print the shape and corresponding label of 3 images in the training set.

for image, label in training_set.take(3):
    print('The images in the training set have:\n\u2022 dtype:', image.dtype, '\n\u2022 shape:', image.shape)


# In[7]:


# TODO: Plot 1 image from the training set. Set the title 
# of the plot to the corresponding image label. 
for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

plt.imshow(image, cmap= plt.cm.binary)
plt.colorbar()
plt.show()

print('The label of this image is:', label)


# ### Label Mapping
# 
# You'll also need to load in a mapping from label to category name. You can find this in the file `label_map.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/3.7/library/json.html). This will give you a dictionary mapping the integer coded labels to the actual names of the flowers.

# In[8]:


import json
with open('label_map.json', 'r') as f:
    class_names = json.load(f)


# In[9]:


print(class_names['52'])


# In[10]:


# TODO: Plot 1 image from the training set. Set the title 
# of the plot to the corresponding class name. 

for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()
    
plt.imshow(image, cmap= plt.cm.binary)
plt.colorbar()
plt.show()

print('The label of this image is:', label)
print('The class name of this image is:', class_names[label.astype('str')])


# ## Create Pipeline

# In[11]:


# TODO: Create a pipeline for each set.
image_size = 224
batch_size = 64

def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image, label

training_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = validation_set.map(format_image).batch(batch_size).prefetch(1)
testing_batches = test_set.map(format_image).batch(batch_size).prefetch(1)


# In[12]:


for image_batch, label_batch in training_batches.take(1):
    print('The images in each batch have:')
    print('\u2022 dtype:', image_batch.dtype) 
    print('\u2022 shape:', image_batch.shape)
  
    print('\nThere are a total of {} image labels in this batch:'.format(label_batch.numpy().size))
    print(label_batch.numpy())


# # Build and Train the Classifier
# 
# Now that the data is ready, it's time to build and train the classifier. You should use the MobileNet pre-trained model from TensorFlow Hub to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! 
# 
# Refer to the rubric for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load the MobileNet pre-trained network from TensorFlow Hub.
# * Define a new, untrained feed-forward network as a classifier.
# * Train the classifier.
# * Plot the loss and accuracy values achieved during training for the training and validation set.
# * Save your trained model as a Keras model. 
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right.
# 
# **Note for Workspace users:** One important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module. Also, If your model is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[13]:


# TODO: Build and train your network.
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size,3))
feature_extractor.trainable = False


# In[14]:


model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(102, activation = 'softmax')
])

model.summary()


# In[15]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 10

history = model.fit(training_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# In[16]:


# TODO: Plot the loss and accuracy values achieved during training for the training and validation set.

training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range=range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[79]:


val_loss, val_acc = model.evaluate(validation_batches)

print('\nLoss on the Validation Set: {}'.format(val_loss))
print('Accuracy on the Validation Set: {}'.format(val_acc))


# ## Testing your Network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[26]:


# TODO: Print the loss and accuracy values achieved on the entire test set.
loss, accuracy = model.evaluate(testing_batches)

print('\nLoss on the TEST Set: {:,.3f}'.format(loss))
print('Accuracy on the TEST Set: {:.3%}'.format(accuracy))


# ## Save the Model
# 
# Now that your network is trained, save the model so you can load it later for making inference. In the cell below save your model as a Keras model (*i.e.* save it as an HDF5 file).

# In[18]:


# TODO: Save your trained model as a Keras model.
model.save('./image_classifier_model.h5')


# ## Load the Keras Model
# 
# Load the Keras model you saved above.

# In[28]:


# TODO: Load the Keras model

reloaded_keras_model = tf.keras.models.load_model('./image_classifier_model.h5',custom_objects={'KerasLayer':hub.KerasLayer})

reloaded_keras_model.summary()


# # Inference for Classification
# 
# Now you'll write a function that uses your trained network for inference. Write a function called `predict` that takes an image, a model, and then returns the top $K$ most likely class labels along with the probabilities. The function call should look like: 
# 
# ```python
# probs, classes = predict(image_path, model, top_k)
# ```
# 
# If `top_k=5` the output of the `predict` function should be something like this:
# 
# ```python
# probs, classes = predict(image_path, model, 5)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# Your `predict` function should use `PIL` to load the image from the given `image_path`. You can use the [Image.open](https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.open) function to load the images. The `Image.open()` function returns an `Image` object. You can convert this `Image` object to a NumPy array by using the `np.asarray()` function.
# 
# The `predict` function will also need to handle pre-processing the input image such that it can be used by your model. We recommend you write a separate function called `process_image` that performs the pre-processing. You can then call the `process_image` function from the `predict` function. 
# 
# ### Image Pre-processing
# 
# The `process_image` function should take in an image (in the form of a NumPy array) and return an image in the form of a NumPy array with shape `(224, 224, 3)`.
# 
# First, you should convert your image into a TensorFlow Tensor and then resize it to the appropriate size using `tf.image.resize`.
# 
# Second, the pixel values of the input images are typically encoded as integers in the range 0-255, but the model expects the pixel values to be floats in the range 0-1. Therefore, you'll also need to normalize the pixel values. 
# 
# Finally, convert your image back to a NumPy array using the `.numpy()` method.

# In[29]:


# TODO: Create the process_image function
def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image


# To check your `process_image` function we have provided 4 images in the `./test_images/` folder:
# 
# * cautleya_spicata.jpg
# * hard-leaved_pocket_orchid.jpg
# * orange_dahlia.jpg
# * wild_pansy.jpg
# 
# The code below loads one of the above images using `PIL` and plots the original image alongside the image produced by your `process_image` function. If your `process_image` function works, the plotted image should be the correct size. 

# In[30]:


from PIL import Image

image_path = './test_images/hard-leaved_pocket_orchid.jpg'
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()


# Once you can get images in the correct format, it's time to write the `predict` function for making inference with your model.
# 
# ### Inference
# 
# Remember, the `predict` function should take an image, a model, and then returns the top $K$ most likely class labels along with the probabilities. The function call should look like: 
# 
# ```python
# probs, classes = predict(image_path, model, top_k)
# ```
# 
# If `top_k=5` the output of the `predict` function should be something like this:
# 
# ```python
# probs, classes = predict(image_path, model, 5)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# Your `predict` function should use `PIL` to load the image from the given `image_path`. You can use the [Image.open](https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.open) function to load the images. The `Image.open()` function returns an `Image` object. You can convert this `Image` object to a NumPy array by using the `np.asarray()` function.
# 
# **Note:** The image returned by the `process_image` function is a NumPy array with shape `(224, 224, 3)` but the model expects the input images to be of shape `(1, 224, 224, 3)`. This extra dimension represents the batch size. We suggest you use the `np.expand_dims()` function to add the extra dimension. 

# In[125]:


# TODO: Create the predict function

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    image = np.asarray(im)
    image = np.expand_dims(process_image(image),axis=0)
    print(image.shape)
    probs = model.predict(image)
    probs, classes = tf.math.top_k(probs, k=top_k, sorted=True)
    return probs.numpy(), classes.numpy()+1
    
    
    


# # Sanity Check
# 
# It's always good to check the predictions made by your model to make sure they are correct. To check your predictions we have provided 4 images in the `./test_images/` folder:
# 
# * cautleya_spicata.jpg
# * hard-leaved_pocket_orchid.jpg
# * orange_dahlia.jpg
# * wild_pansy.jpg
# 
# In the cell below use `matplotlib` to plot the input image alongside the probabilities for the top 5 classes predicted by your model. Plot the probabilities as a bar graph. The plot should look like this:
# 
# <img src='assets/inference_example.png' width=600px>
# 
# You can convert from the class integer labels to actual flower names using `class_names`. 

# In[129]:


# TODO: Plot the input image along with the top 5 classes

top_k = 5
image_path = './test_images/hard-leaved_pocket_orchid.jpg'
im = Image.open(image_path)
image = np.asarray(im)
processed_image = process_image(image)
probs, classes = predict(image_path, model, top_k)
print(probs[0])
print(classes[0])




# In[127]:


classnames= [class_names[i.astype('str')] for i in classes[0,:]]


# In[128]:


fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
ax1.imshow(processed_image, cmap = plt.cm.binary)
ax1.axis('off')
y_pos = np.arange(len(classnames))
ax2.barh(y_pos, probs[0], align='center', color='blue')
ax2.set_aspect(0.1)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(classnames, size='small');
ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)
plt.tight_layout()

