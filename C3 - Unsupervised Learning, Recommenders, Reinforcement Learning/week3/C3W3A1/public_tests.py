from tensorflow.keras.activations import relu, linear
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np

def test_network(target):
    num_actions = 4
    state_size = 8
    i = 0
    assert len(target.layers) == 3, f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.input.shape.as_list() == [None, state_size], \
        f"Wrong input shape. Expected [None,  400] but got {target.input.shape.as_list()}" 
    expected = [[Dense, [None, 64], relu],
                [Dense, [None, 64], relu],
                [Dense, [None, num_actions], linear]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        i = i + 1

    print("\033[92mAll tests passed!")
    
def test_optimizer(target, ALPHA):
    assert type(target) == Adam, f"Wrong optimizer. Expected: {Adam}, got: {target}"
    assert np.isclose(target.learning_rate.numpy(), ALPHA), f"Wrong alpha. Expected: {ALPHA}, got: {target.learning_rate.numpy()}"
    print("\033[92mAll tests passed!")
    
    
def test_compute_loss(target):
    num_actions = 4
    def target_q_network_random(inputs):
        return np.float32(np.random.rand(inputs.shape[0],num_actions))
    
    def q_network_random(inputs):
        return np.float32(np.random.rand(inputs.shape[0],num_actions))
    
    def target_q_network_ones(inputs):
        return np.float32(np.ones((inputs.shape[0], num_actions)))
    
    def q_network_ones(inputs):
        return np.float32(np.ones((inputs.shape[0], num_actions)))
    
    np.random.seed(1)
    states = np.float32(np.random.rand(64, 8))
    actions = np.float32(np.floor(np.random.uniform(0, 1, (64, )) * 4))
    rewards = np.float32(np.random.rand(64, ))
    next_states = np.float32(np.random.rand(64, 8))
    done_vals = np.float32((np.random.uniform(0, 1, size=(64,)) > 0.96) * 1)

    loss = target((states, actions, rewards, next_states, done_vals), 0.995, q_network_random, target_q_network_random)
    

    assert np.isclose(loss, 0.6991737), f"Wrong value. Expected {0.6991737}, got {loss}"

    # Test when episode terminates
    done_vals = np.float32(np.ones((64,)))
    loss = target((states, actions, rewards, next_states, done_vals), 0.995, q_network_ones, target_q_network_ones)
    assert np.isclose(loss, 0.343270182), f"Wrong value. Expected {0.343270182}, got {loss}"
      
    # Test MSE with parameters A = B
    done_vals = np.float32((np.random.uniform(0, 1, size=(64,)) > 0.96) * 1)
    rewards = np.float32(np.ones((64, )))
    loss = target((states, actions, rewards, next_states, done_vals), 0, q_network_ones, target_q_network_ones)
    assert np.isclose(loss, 0), f"Wrong value. Expected {0}, got {loss}"
 
    # Test MSE with parameters A = 0 and B = 1
    done_vals = np.float32((np.random.uniform(0, 1, size=(64,)) > 0.96) * 1)
    rewards = np.float32(np.zeros((64, )))
    loss = target((states, actions, rewards, next_states, done_vals), 0, q_network_ones, target_q_network_ones)
    assert np.isclose(loss, 1), f"Wrong value. Expected {1}, got {loss}"

    print("\033[92mAll tests passed!")
    