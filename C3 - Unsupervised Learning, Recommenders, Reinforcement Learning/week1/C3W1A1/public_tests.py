import numpy as np

def compute_centroids_test(target):
    # With 3 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [-1.1, -1.7], [-1.6, 1.2]])
    idx = np.array([1, 1, 1, 0, 0, 0, 2])
    K = 3
    centroids = target(X, idx, K)
    expected_centroids = np.array([[0.13333333,  0.43333333],
                                   [-1.33333333, -0.5      ],
                                   [-1.6,        1.2       ]])
    
    assert type(centroids) == np.ndarray, "Wrong type"
    assert centroids.shape == (K, X.shape[1]), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(centroids, expected_centroids), f"Wrong values. Expected: {expected_centroids}, got: {centroids}"
    
    X = np.array([[2, 2.5], [2.5, 2.5], [-1.5, -1.5],
                  [2, 2], [-1.5, -1], [-1, -1]])
    idx = np.array([0, 0, 1, 0, 1, 1])
    K = 2
    centroids = target(X, idx, K)
    expected_centroids = np.array([[[ 2.16666667,  2.33333333],
                                    [-1.33333333, -1.16666667]]])
    
    assert type(centroids) == np.ndarray, "Wrong type"
    assert centroids.shape == (K, X.shape[1]), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(centroids, expected_centroids), f"Wrong values. Expected: {expected_centroids}, got: {centroids}"
    
    print("\033[92mAll tests passed!")
    
def find_closest_centroids_test(target):
    # With 2 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, -1],
                  [2, 2],[2.5, 2.5],[2, 2.5]])
    initial_centroids = np.array([[-1, -1], [2, 2]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "Wrong type"
    assert idx.shape == (len(X),), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(idx, [0, 0, 0, 1, 1, 1]), "Wrong values"
    
    # With 3 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [2, 2]])
    initial_centroids = np.array([[2.5, 2], [-1, -1], [-1.5, 1.]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "Wrong type"
    assert idx.shape == (len(X),), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(idx, [1, 1, 2, 2, 0, 0]), f"Wrong values. Expected {[2, 2, 0, 0, 1, 1]}, got: {idx}"
    
    # With 3 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [-1.1, -1.7], [-1.6, 1.2]])
    initial_centroids = np.array([[2.5, 2], [-1, -1], [-1.5, 1.]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "Wrong type"
    assert idx.shape == (len(X),), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(idx, [1, 1, 2, 2, 0, 1, 2]), f"Wrong values. Expected {[2, 2, 0, 0, 1, 1]}, got: {idx}"
    
    print("\033[92mAll tests passed!")