"""
Author: Riccardo Andreoni
Title: Comparison between different optimizers on the Fashion MNIST classification
problem.
File: main.py
"""

import matplotlib.pyplot as plt
import tensorflow as tf

def load_my_data():
    """
    Returns the fashin MNIST dataset split into train and test sets.
    """
    # Load data from the keras datasets
    fmnist = tf.keras.datasets.fashion_mnist
    # Split the data in train and test sets
    (x_train, y_train), (x_test, y_test) = fmnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    
    return (x_train, y_train), (x_test, y_test)


def main():
    (x_train, y_train), (x_test, y_test) = load_my_data()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()

