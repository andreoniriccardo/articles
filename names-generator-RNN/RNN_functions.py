import numpy as np
import copy

"""
Arguments
---------

Returns
-------
"""

def initialize_parameters(n_a, n_x, n_y):
    """
    Initialize parameters with small random values. Generate random values and keep them
    small by multiplying by 0.01.
    
    Arguments
    ---------
    n_a (int): number of units of the hidden state
    n_x (int): input's vocabulary size
    n_y (int): outputs's vocabulary size

    Returns
    -------
    parameters -- python dictionary containing:
        W_ax -- weight matrix multiplying the input, np.array of shape (n_a, n_x)
        W_aa -- weight matrix multiplying the hidden state, np.array of shape (n_a, n_a)
        W_ya -- weight matrix relating the hidden-state to the output, np.array of shape (n_y, n_a)
        b --  bias, np.array of shape (n_a, 1)
        b_y -- bias relating the hidden-state to the output, np.array of shape (n_y, 1)
    """

    keep_small = 0.01

    W_ax = np.random.randn(n_a, n_x)*keep_small
    W_aa = np.random.randn(n_a, n_a)*keep_small
    W_ya = np.random.randn(n_y, n_a)*keep_small
    b = np.zeros((n_a, 1))
    b_y = np.zeros((n_y, 1))
    
    parameters = {"W_ax": W_ax, 
                  "W_aa": W_aa, 
                  "W_ya": W_ya, 
                  "b": b,
                  "b_y": b_y}
    
    return parameters

def get_initial_loss(vocab_size, n_examples):
    """
    Initialize the loss function to smooth the loss value.

    Arguments
    ---------
    vocab_size (int): input's vocabulary size
    n_examples (int): number of instances in the training dataset

    Returns
    -------
    initial loss function (float)

    """
    return -np.log(1.0/vocab_size)*n_examples

def initialize_hidden_state(n_a):
    """
    Initialize hidden state vector values as zeros.

    Arguments
    ---------
    n_a (int): number of units of the hidden state

    Returns
    -------
    np.array of zeros

    """
    return np.zeros((n_a, 1))

def encode_chars(chars):
    chars = sorted(chars)
    return { ch:i for i,ch in enumerate(chars) }

def decode_chars(chars):
    return { i:ch for i,ch in enumerate(chars) }

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def smooth(loss_now, loss_tmp):
    return loss_now * 0.999 + loss_tmp * 0.001

def clip_grads(grads, max_value):
    grads = copy.deepcopy(grads)
    dW_aa, dW_ax, dW_ya, db, db_y = grads['dW_aa'], grads['dW_ax'], grads['dW_ya'], grads['db'], grads['db_y']

    for grads in [dW_ax, dW_aa, dW_ya, db, db_y]:
        np.clip(grads, -max_value, max_value, out = grads)

    grads = {"dW_aa": dW_aa, "dW_ax": dW_ax, "dW_ya": dW_ya, "db": db, "db_y": db_y}
    
    return grads


def update_parameters(parameters, grads, lr):

    parameters['W_ax'] += -lr * grads['dW_ax']
    parameters['W_aa'] += -lr * grads['dW_aa']
    parameters['W_ya'] += -lr * grads['dW_ya']
    parameters['b']  += -lr * grads['db']
    parameters['b_y']  += -lr * grads['db_y']
    return parameters


def RNN_forward_prop_step(parameters, a_prev, x):
    W_aa = parameters['W_aa']
    W_ax = parameters['W_ax']
    W_ya = parameters['W_ya']
    b_y = parameters['b_y']
    b = parameters['b']

    # Compute hidden state
    a_next = np.tanh(np.dot(W_ax, x) + np.dot(W_aa, a_prev) + b)
    part_1 = np.dot(W_ax, x)
    
    # Compute log probabilities for next character
    p_t = softmax(np.dot(W_ya, a_next) + b_y)

    return a_next, p_t

def RNN_forward_prop(X, Y, a0, parameters, vocab_size):

    x = {}
    a = {}
    y_hat = {}
    a[-1] = np.copy(a0)
    loss= 0

    # Iterate for the T timesteps
    for t in range(len(X)):

        # One-hot representation of the t-th character
        # If the value is None (first step), x[t] is a zero vector 
        x[t] = np.zeros((vocab_size, 1))
        if X[t] != None:
            x[t][X[t]] = 1

        # Run one timestep of forward prop
        # print(a[t-1])
        a[t], y_hat[t] = RNN_forward_prop_step(parameters, a[t-1], x[t])

        # Update loss function
        loss -= np.log(y_hat[t][Y[t],0])

    cache = (y_hat, a, x)

    return loss, cache

def RNN_back_prop_step(d_y, grads, parameters, x, a, a_prev):
    
    grads['dW_ya'] += np.dot(d_y, a.T)
    grads['db_y'] += d_y
    da = np.dot(parameters['W_ya'].T, d_y) + grads['da_next']
    daraw = (1 - a * a) * da
    grads['db'] += daraw
    grads['dW_ax'] += np.dot(daraw, x.T)
    grads['dW_aa'] += np.dot(daraw, a_prev.T)
    grads['da_next'] = np.dot(parameters['W_aa'].T, daraw)
    return grads

def RNN_back_prop(X, Y, parameters, cache):
    # Initialize gradients as an empty dictionary
    grads = {}
    
    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    W_aa = parameters['W_aa']
    W_ax = parameters['W_ax']
    W_ya = parameters['W_ya']
    b_y = parameters['b_y']
    b = parameters['b']


    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    grads['dW_ax'], grads['dW_aa'], grads['dW_ya'] = np.zeros_like(W_ax), np.zeros_like(W_aa), np.zeros_like(W_ya)
    grads['db'], grads['db_y'] = np.zeros_like(b), np.zeros_like(b_y)
    grads['da_next'] = np.zeros_like(a[0])
    
    # Backpropagate through timesteps
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        grads = RNN_back_prop_step(dy, grads, parameters, x[t], a[t], a[t-1])
        
    return grads, a



def RNN_optimization(X, Y, a_prev, parameters, alpha, vocab_size):
    """
    Perform one step of the optimization cycle:
        1. Forward propagation
        2. Backward propagation
        3. Gradient clipping
        4. Parameters update
    
    Arguments
    ---------
    
    Returns
    -------
      
    
    """

    # 1. Forward propagation
    loss_now, cache = RNN_forward_prop(X, Y, a_prev, parameters, vocab_size)

    # 2. Backward propagation
    grads, a = RNN_back_prop(X, Y, parameters, cache)

    # 3. Clip gradients
    grads = clip_grads(grads, 10)

    # 4. Update parameters
    parameters = update_parameters(parameters, grads, alpha)

    return loss_now, parameters, a[len(X)-1]

def sample(parameters, chars_to_encoding):
    # Retrieve from cache and parameters
    W_aa = parameters['W_aa']
    W_ax = parameters['W_ax']
    W_ya = parameters['W_ya']
    b_y = parameters['b_y']
    b = parameters['b']
    vocab_size = b_y.shape[0]
    n_a = W_aa.shape[1]

    x = np.zeros((vocab_size,))
    a_prev = np.zeros((n_a,))

    indices = []
    idx = -1 

    counter = 0
    newline_character = chars_to_encoding['\n']

    while (idx != newline_character and counter != 50):
        a = np.tanh(np.dot(W_ax,x)+np.dot(W_aa,a_prev)+np.ravel(b))
        z = np.dot(W_ya,a) + np.ravel(b_y)
        y = softmax(z)

        idx = np.random.choice(list(chars_to_encoding.values()), p=np.ravel(y))
        indices.append(idx)    

        x = np.zeros((vocab_size,))
        x[idx] = 1

        a_prev = a

        counter +=1
        if (counter == 50):
            indices.append(chars_to_encoding['\n'])
    return indices

def get_sample(sample_ix, encoding_to_chars):
    txt = ''.join(encoding_to_chars[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character 
    return txt

def train_model(data, n_a=50, max_iter = 40000):
    """
    Trains the language model to generate novel fictional character names.

    Arguments
    ---------
    data (str): training set containing the actual LOTR characters names. Each line contains a name.
    n_a (int): number of units of the hidden state

    Returns
    -------
    parameters (dict): dictionary containing trained model's parameters

    """
    # Get the list of characters
    chars = list(set(data))
    # Get the dictionary size (number of characters)
    vocab_size = len(chars)

    # Get encoding and decoding dictionaries
    chars_to_encoding = encode_chars(chars)
    encoding_to_chars = decode_chars(chars)

    # Get dataset as a list of names and strip, then shuffle the dataset
    data = data.split('\n')
    data = [x.strip() for x in data]   
    np.random.shuffle(data) 

    # Define n_x, n_y parameters
    n_x, n_y = vocab_size, vocab_size

    # Initialize the hidden state
    # a_prev = initialize_hidden_state(n_a)
    a_prev = np.zeros((n_a, 1))

    # Initialize the parameters
    parameters = initialize_parameters(n_a, n_x, n_y)
    # for k in parameters.keys():
    #     print('{}: tipo {}, Datatype {}'.format(k, type(parameters[k]), parameters[k].dtype))
    # Get current loss function value
    loss_now = get_initial_loss(vocab_size, len(data))

    loss_array = []
    loss_array.append(loss_now)

    # Perform max_iter iteration to train the model's parameters
    for iter in range(max_iter):
        # print(iter)
        # Get the index of the name to pick
        name_idx = iter % len(data)
        example = data[name_idx]

        # Convert encoded and decoded example into a list
        example_chars = [char for char in example]
        example_encoded = [chars_to_encoding[char] for char in example]

        # Create training input X. The value None is used to consider the first input character
        # as a vector of zeros
        X = [None] + example_encoded

        # Create the label vector Y by appending the '\n' encoding to the end of the vector
        Y = example_encoded + [chars_to_encoding['\n']]

        # Perform one step of the optimization cycle:
        # 1. Forward propagation
        # 2. Backward propagation
        # 3. Gradient clipping
        # 4. Parameters update

        loss_tmp, parameters, a_prev = RNN_optimization(X, Y, a_prev, parameters, alpha=0.01, vocab_size=vocab_size)
        # for k in parameters.keys():
        #     print('{}: tipo {}, Datatype {}'.format(k, type(parameters[k]), parameters[k].dtype))
        loss_now = smooth(loss_now, loss_tmp)
        loss_array.append(loss_now)

        # For every 1000 gens, print loss and generate names:
        if iter % 1000 == 0:
            print('Iteration {}, Loss: {}\n'.format(iter, loss_now))

            sampled_indices = sample(parameters, chars_to_encoding)
            last_fantasy_name = get_sample(sampled_indices, encoding_to_chars)
            print(last_fantasy_name.replace('\n', ''))

    return parameters, loss_array