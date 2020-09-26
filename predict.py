# Import TensorFlow 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# Make all other necessary imports.
import matplotlib.pyplot as plt
import json
import numpy as np

from PIL import Image
import argparse

batch_size = 64
image_size = 224

def process_image(image):
    image = tf.cast(image, tf.float32)
    # you can also do this for conversion tf.image.convert_image_dtype(x, dtype=tf.float16, saturate=False)
    image = tf.image.resize(image, (image_size, image_size)).numpy()
    image /= 255
    
    return image


def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    test_image = np.asarray(image)
    processed_test_image = process_image(test_image)
    expanded_test_image = np.expand_dims(processed_test_image, axis=0)
    
    pred_image = model.predict(expanded_test_image)
    values, indices = tf.math.top_k(pred_image, k=top_k)
    probs = values.numpy()[0]
    classes = indices.numpy()[0] + 1
    
    # preapere the result for presenting
    probs = list(probs)
    classes = list(map(str, classes))
    
    return probs, classes


class_names = [ ]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names') 
    
    
    args = parser.parse_args()
    
    image_path = args.arg1
    
    model = tf.keras.models.load_model(args.arg2 ,custom_objects={'KerasLayer':hub.KerasLayer})
    
    """
    Basic usage command:
           python predict.py /path/to/image saved_model
    """
    if args.top_k is None and args.category_names is None:
        probs, classes = predict(image_path, model)
        print("The probabilities and classes of the images: ")
        
    """
    Options:
           python predict.py /path/to/image saved_model --top_k K
           
           --top_k: Return the top K most likely classes:
    """
    elif args.top_k is not None:
        top_k = int(args.top_k)
        probs, classes = predict(image_path, model, top_k)
        print("The top {} probabilities and classes of the images: ".format(top_k))
       
    """
    Options:
           python predict.py /path/to/image saved_model --category_names map.json
           
           --category_names: Path to a JSON file mapping labels to flower names:
    """
    elif args.category_names is not None:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        probs, classes = predict(image_path, model)
        print("The probabilities and classes of the images: ")
        classes = [class_names[class_] for class_ in  classes]
        
            
            
    for prob, class_ in zip(probs, classes):
        print('\u2022 "{}":  {:.3%}'.format(class_, prob))
        
    
    print('\nThe flower is: "{}"'.format(classes[0]))
    
    
    