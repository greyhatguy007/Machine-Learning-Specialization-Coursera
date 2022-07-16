# UNIT TESTS
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense

import numpy as np

def test_c1(target):
    assert len(target.layers) == 3, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.input.shape.as_list() == [None, 400], \
        f"Wrong input shape. Expected [None,  400] but got {target.input.shape.as_list()}"
    i = 0
    expected = [[Dense, [None, 25], sigmoid],
                [Dense, [None, 15], sigmoid],
                [Dense, [None, 1], sigmoid]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        i = i + 1

    print("\033[92mAll tests passed!")
    
def test_c2(target):
    
    def linear(a):
        return a
    
    def linear_times3(a):
        return a * 3
    
    x_tst = np.array([1., 2., 3., 4.])  # (1 examples, 3 features)
    W_tst = np.array([[1., 2.], [1., 2.], [1., 2.], [1., 2.]]) # (3 input features, 2 output features)
    b_tst = np.array([0., 0.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert A_tst.shape[0] == len(b_tst)
    assert np.allclose(A_tst, [10., 20.]), \
        "Wrong output. Check the dot product"
    
    b_tst = np.array([3., 5.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert np.allclose(A_tst, [13., 25.]), \
        "Wrong output. Check the bias term in the formula"
    
    A_tst = target(x_tst, W_tst, b_tst, linear_times3)
    assert np.allclose(A_tst, [39., 75.]), \
        "Wrong output. Did you apply the activation function at the end?"
    
    print("\033[92mAll tests passed!")  
    
def test_c3(target):
    
    def linear(a):
        return a
    
    def linear_times3(a):
        return a * 3
    
    x_tst = np.array([1., 2., 3., 4.])  # (1 examples, 3 features)
    W_tst = np.array([[1., 2.], [1., 2.], [1., 2.], [1., 2.]]) # (3 input features, 2 output features)
    b_tst = np.array([0., 0.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert A_tst.shape[0] == len(b_tst)
    assert np.allclose(A_tst, [10., 20.]), \
        "Wrong output. Check the dot product"
    
    b_tst = np.array([3., 5.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert np.allclose(A_tst, [13., 25.]), \
        "Wrong output. Check the bias term in the formula"
    
    A_tst = target(x_tst, W_tst, b_tst, linear_times3)
    assert np.allclose(A_tst, [39., 75.]), \
        "Wrong output. Did you apply the activation function at the end?"
    
    x_tst = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])  # (2 examples, 4 features)
    W_tst = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12]]) # (3 input features, 2 output features)
    b_tst = np.array([0., 0., 0.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert A_tst.shape == (2, 3)
    assert np.allclose(A_tst, [[ 70.,  80.,  90.], [158., 184., 210.]]), \
        "Wrong output. Check the dot product"
    
    b_tst = np.array([3., 5., 6])  # (3 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert np.allclose(A_tst, [[ 73.,  85.,  96.], [161., 189., 216.]]), \
        "Wrong output. Check the bias term in the formula"
    
    A_tst = target(x_tst, W_tst, b_tst, linear_times3)
    assert np.allclose(A_tst, [[ 219.,  255.,  288.], [483., 567., 648.]]), \
        "Wrong output. Did you apply the activation function at the end?"
    
    print("\033[92mAll tests passed!")  
