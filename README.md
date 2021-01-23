# Image Classifier - TensorFlow Project
In this project, we will first develop code for an image classifier built with TensorFlow, then we will convert it into a command-line application.

We'll be using [this dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from Oxford of 102 flower categories. The data for this project is quite large - in fact, it is so large that you cannot upload it onto Github.

# Requirements
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)


Before you run any commands in the terminal make sure to install TensorFlow 2.0 and TensorFlow Hub using ```pip``` as shown below:
```
$ pip install -q -U "tensorflow-gpu==2.0.0b1"
```
```
$ pip install -q -U tensorflow_hub
```


*Note*: In order to complete this project, you will need to use the GPU. since running on your local CPU will likely not work well. You should also only enable the GPU when you need it.  

# Project Structure
The project is constituted by two parts:

## Part 1 - Developing an Image Classifier with Deep Learning
This part of the project is broken down into multiple steps:
1. Load the image dataset and create a pipeline.
2. Build and Train an image classifier on this dataset.
3. Use your trained model to perform inference on flower images.
4. Save the model, for later applications.
   
## Part 2 - Building the Command Line Application
After we have built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. The application is a Python script that runs from the command line. For testing, we would use the saved Keras model that has been saved in the first part.

### Specifications
The [**predict.py file**](predict.py) uses [the saved trained network](my_model.h5) to predict the class for an input image. It should predict the top flower names from an image along with their corresponding probabilities also.

**Basic usage:**
```
$ python predict.py /path/to/image saved_model
```
**Options:**
- ```--top_k``` : Return the top **K** most likely classes:
```
$ python predict.py /path/to/image saved_model --top_k K
```
- ```--category_names``` : Path to a JSON file mapping labels to flower
names:
```
$ python predict.py /path/to/image saved_model --category_names map.json
```

#### Example:

 Assume that we have a file called ```orchid.jpg``` in a folder named ```/test_images/``` that contains the image of a flower. We also assume that we have a Keras model saved in a file named ```my_model.h5```. 

 **Basic usage:**
 ```
 $ python predict.py ./test_images/orchid.jpg my_model.h5
 ```

**Options:**
- Return the top 3 most likely classes:
```
$ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
```

 - Assume we use a ```label_map.json``` file to map labels to flower names: 
  ```
  $ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
  ```

# Images for Testing
In the Command Line Interface workspace, there are 4 images provided in the [```./test_images/ folder```](test_images/) for you to check your ```predict.py``` module. The 4 images are:
- cautleya_spicata.jpg
- hard-leaved_pocket_orchid.jpg
- orange_dahlia.jpg
- wild_pansy.jpg





