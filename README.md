# cat-and-notcats
Neural network that identifies whether an image is a cat or not

## Libraries
- TensorFlow (GPU)
- Keras
- PIL
- numpy

## How to Use
For arg_test.py:
```
python arg_test.py -m [MODEL] -i [IMAGE]
```

For rand_test.py
```
python rand_test.py -m [MODEL] -d [DIRECTORY]
```

Don't run main.py unless you're training a new model all together. Run retrain.py to keep training the model.

## References:
- https://www.codeproject.com/Articles/4023566/Cat-or-Not-An-Image-Classifier-using-Python-and-Ke
- https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
- https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
- https://medium.com/@curiousily/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b

## Datasets Used:
- https://www.kaggle.com/c/dogs-vs-cats
- My own cat pictures
