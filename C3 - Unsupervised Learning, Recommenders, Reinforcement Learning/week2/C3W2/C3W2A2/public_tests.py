from tensorflow.keras.activations import relu, linear
from tensorflow.keras.layers import Dense

import numpy as np

def test_tower(target):
    num_outputs = 32
    i = 0
    assert len(target.layers) == 3, f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    expected = [[Dense, [None, 256], relu],
                [Dense, [None, 128], relu],
                [Dense, [None, num_outputs], linear]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        i = i + 1

    print("\033[92mAll tests passed!")


def test_sq_dist(target):
    a1 = np.array([1.0, 2.0, 3.0]); b1 = np.array([1.0, 2.0, 3.0])
    c1 = target(a1, b1)
    a2 = np.array([1.1, 2.1, 3.1]); b2 = np.array([1.0, 2.0, 3.0])
    c2 = target(a2, b2)
    a3 = np.array([0, 1]);          b3 = np.array([1, 0])
    c3 = target(a3, b3)
    a4 = np.array([1, 1, 1, 1, 1]); b4 = np.array([0, 0, 0, 0, 0])
    c4 = target(a4, b4)
    
    assert np.isclose(c1, 0), f"Wrong value. Expected {0}, got {c1}"
    assert np.isclose(c2, 0.03), f"Wrong value. Expected {0.03}, got {c2}" 
    assert np.isclose(c3, 2), f"Wrong value. Expected {2}, got {c3}" 
    assert np.isclose(c4, 5), f"Wrong value. Expected {5}, got {c4}" 
    
    print('\033[92mAll tests passed!')
