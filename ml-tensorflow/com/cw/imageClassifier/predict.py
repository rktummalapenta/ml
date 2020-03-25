import sys
import json
import argparse
import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image

def predict(image_path, model_path, top_k):
    im = Image.open(image_path)
    image = np.asarray(im)
    image = np.expand_dims(process_image(image),axis=0)
    model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    probs = model.predict(image)
    probs, classes = tf.math.top_k(probs, k=top_k, sorted=True)
    return probs.numpy(), classes.numpy()+1


def main():
    # Create the parser and add the arguments
    parser = argparse.ArgumentParser(description="Load a Neural Network to use for inference")
    parser.add_argument('positionals', nargs='+')
    parser.add_argument('--top_k', default=1, type=int,
                    help="The amount of most likley classes to return for the predictions")
    parser.add_argument('--category_names', default = './label_map.json',
                    help="The json file (including file path) to load category names")

    # Collect the arguments
    args = parser.parse_args()
    return args
   
if __name__ == "__main__":
    print('Using:')
    print('\t\u2022 TensorFlow version:', tf.__version__)
    print('\t\u2022 tf.keras version:', tf.keras.__version__)
    print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')
    args = main()
    image_path = args.positionals[0]
    model_path = args.positionals[1]
    top_k = args.top_k
    category_name = args.category_names
    image_size = 224
    # Load the content of the json file
    f = open(category_name,)
    class_names = json.load(f)
    probs, classes = predict(image_path,model_path,top_k)
    prob=[ i for i in probs[0,:]]
    classnames= [class_names[i.astype('str')] for i in classes[0,:]]
    print('The most likely class names list: {} and its associated probabilities list:{} '.format(classnames,prob))
    