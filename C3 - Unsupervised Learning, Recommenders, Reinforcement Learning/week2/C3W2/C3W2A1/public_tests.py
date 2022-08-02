import numpy as np

def test_cofi_cost_func(target):
    num_users_r = 4
    num_movies_r = 5 
    num_features_r = 3

    X_r = np.ones((num_movies_r, num_features_r))
    W_r = np.ones((num_users_r, num_features_r))
    b_r = np.zeros((1, num_users_r))
    Y_r = np.zeros((num_movies_r, num_users_r))
    R_r = np.zeros((num_movies_r, num_users_r))
    
    J = target(X_r, W_r, b_r, Y_r, R_r, 2);
    assert not np.isclose(J, 13.5), f"Wrong value. Got {J}. Did you multiplied the regulartization term by lambda_?"
    assert np.isclose(J, 27), f"Wrong value. Expected {27}, got {J}. Check the regularization term"
    
    
    X_r = np.ones((num_movies_r, num_features_r))
    W_r = np.ones((num_users_r, num_features_r))
    b_r = np.ones((1, num_users_r))
    Y_r = np.ones((num_movies_r, num_users_r))
    R_r = np.ones((num_movies_r, num_users_r))

    # Evaluate cost function
    J = target(X_r, W_r, b_r, Y_r, R_r, 0);
    
    assert np.isclose(J, 90), f"Wrong value. Expected {90}, got {J}. Check the term without the regularization"
    
    
    X_r = np.ones((num_movies_r, num_features_r))
    W_r = np.ones((num_users_r, num_features_r))
    b_r = np.ones((1, num_users_r))
    Y_r = np.zeros((num_movies_r, num_users_r))
    R_r = np.ones((num_movies_r, num_users_r))

    # Evaluate cost function
    J = target(X_r, W_r, b_r, Y_r, R_r, 0);
    
    assert np.isclose(J, 160), f"Wrong value. Expected {160}, got {J}. Check the term without the regularization"
    
    X_r = np.ones((num_movies_r, num_features_r))
    W_r = np.ones((num_users_r, num_features_r))
    b_r = np.ones((1, num_users_r))
    Y_r = np.ones((num_movies_r, num_users_r))
    R_r = np.ones((num_movies_r, num_users_r))

    # Evaluate cost function
    J = target(X_r, W_r, b_r, Y_r, R_r, 1);
    
    assert np.isclose(J, 103.5), f"Wrong value. Expected {103.5}, got {J}. Check the term without the regularization"
    
    num_users_r = 3
    num_movies_r = 4 
    num_features_r = 4
    
    #np.random.seed(247)
    X_r = np.array([[0.36618032, 0.9075415,  0.8310605,  0.08590986],
                     [0.62634721, 0.38234325, 0.85624346, 0.55183039],
                     [0.77458727, 0.35704147, 0.31003294, 0.20100006],
                     [0.34420469, 0.46103436, 0.88638208, 0.36175401]])#np.random.rand(num_movies_r, num_features_r)
    W_r = np.array([[0.04786854, 0.61504665, 0.06633146, 0.38298908], 
                    [0.16515965, 0.22320207, 0.89826005, 0.14373251], 
                    [0.1274051 , 0.22757303, 0.96865613, 0.70741111]])#np.random.rand(num_users_r, num_features_r)
    b_r = np.array([[0.14246472, 0.30110933, 0.56141144]])#np.random.rand(1, num_users_r)
    Y_r = np.array([[0.20651685, 0.60767914, 0.86344527], 
                    [0.82665019, 0.00944765, 0.4376798 ], 
                    [0.81623732, 0.26776794, 0.03757507], 
                    [0.37232161, 0.19890823, 0.13026598]])#np.random.rand(num_movies_r, num_users_r)
    R_r = np.array([[1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]])#(np.random.rand(num_movies_r, num_users_r) > 0.4) * 1

    # Evaluate cost function
    J = target(X_r, W_r, b_r, Y_r, R_r, 3);
    
    assert np.isclose(J, 13.621929978531858, atol=1e-8), f"Wrong value. Expected {13.621929978531858}, got {J}."
    
    print('\033[92mAll tests passed!')
    
