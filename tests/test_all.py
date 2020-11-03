import numpy as np
from rapport import *

def test_mnist_length():
    mnist_data = np.zeros((100,50))
    mnist_target =  np.zeros((100,50))
    dataset_length, target_length = data_length(mnist_data, mnist_target)
    assert dataset_length == 100
    assert target_length == 100

def test_normalize():
    arr =  np.arange(6).reshape(2,-1)
    arr_test = arr / 255
    assert np.allclose(normalize_data(arr), arr_test)

def test_one_hot():
    arr =  np.array([0,1,9])
    arr_test = np.array([[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1]])
    assert np.allclose(target_to_one_hot(arr), arr_test)

def test_sigmoid():
    arr = np.arange(6).reshape(2,-1)
    arr_test = np.array([[0.5, 0.73105858, 0.88079708], [0.95257413, 0.98201379, 0.99330715]])
    assert np.allclose(sigmoid(arr), arr_test)

def test_dsigmoid():
    arr = np.arange(6).reshape(2,-1)
    arr_test = np.array([[0.25, 0.19661193, 0.10499359], [0.04517666, 0.01766271, 0.00664806]])
    assert np.allclose(d_sigmoid(arr), arr_test)

def test_softmax():
    arr = np.arange(6).reshape(2,-1)
    arr_test = np.array([[0.09003057, 0.24472847, 0.66524096],[0.09003057, 0.24472847, 0.66524096]])
    assert np.allclose(softmax(arr), arr_test)

def test_ffnn():
    ffnn = FFNN(config=[784, 3, 3, 10], minibatch_size=5, learning_rate=0.05)
    arr_pred = np.array([[0,0,1],[1,0,0]])
    arr_true = np.array([[0,0,1],[0,1,0]])
    assert ffnn.get_error(arr_pred, arr_true) == 0.5
