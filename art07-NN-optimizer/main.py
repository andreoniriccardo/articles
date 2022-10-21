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

def print_image(dataset, index=0):
    plt.imshow(dataset[index])
    plt.show()
    


def main():
    # Load data
    (X_train, y_train), (X_test, y_test) = load_my_data()
    # Print an example
    #print_image(X_train, 42)
    
    # Normalize the inputs
    X_train = X_train / 255.
    X_test = X_test / 255.
    
       
    # Define the optimizers to test
    my_optimizers = {"Mini-batch GD":tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.0),
                     "Momentum GD":tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
                     "RMS Prop":tf.keras.optimizers.RMSprop(learning_rate = 0.01, rho = 0.9),
                     "Adam":tf.keras.optimizers.Adam(learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.999)
        }
    
    histories = []
    for optimizer_name, optimizer in my_optimizers.items():
        # Define a neural network
        my_network = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation='elu'),
                    tf.keras.layers.Dense(256, activation='elu'),
                    tf.keras.layers.Dense(128, activation='elu'),
                    tf.keras.layers.Dense(64, activation='elu'),
                    tf.keras.layers.Dense(10, activation='softmax')
                    ])
        
        # Compile the model
        my_network.compile(optimizer=optimizer,
                           loss='sparse_categorical_crossentropy', # since labels are more than 2 and not one-hot-encoded
                           metrics=['accuracy'])
    
        # Train the model
        print('Training the model with optimizer {}'.format(optimizer_name))
        history = my_network.fit(X_train, y_train, epochs=8, validation_split=0.1, verbose=1)
        histories.append(history)
    
        
    
if __name__ == '__main__':
    main()

