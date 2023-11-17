import numpy as np

from tree_utils import LeafNode, InternalDecisionNode

def calculate_gini_impurity(y_G):
    ''' Calculate Gini impurity for a given set of labels y.'''
    if len(y_G) == 0:
        return 0.0
    p_1 = np.sum(y_G == 1) / len(y_G)
    p_0 = 1 - p_1
    gini = 1 - (p_0**2 + p_1**2)
    return gini


def select_best_binary_split(x_NF, y_N, MIN_SAMPLES_LEAF=1):
    ''' Determine best single feature binary split for provided dataset

    Args
    ----
    x_NF : 2D array, shape (N,F) = (n_examples, n_features)
        Training data features at current node we wish to find a split for.
    y_N : 1D array, shape (N,) = (n_examples,)
        Training labels at current node.
    min_samples_leaf : int
        Minimum number of samples allowed at any leaf.

    Returns
    -------
    feat_id : int or None, one of {0, 1, 2, .... F-1}
        Indicates which feature in provided x array is used for best split.
        If None, a binary split that improves the cost is not possible.
    thresh_val : float or None
        Value of x[feat_id] at which we threshold.
        If None, a binary split that improves the cost is not possible.
    x_LF : 2D array, shape (L, F)
        Training data features assigned to left child using best split.
    y_L : 1D array, shape (L,)
        Training labels assigned to left child using best split.
    x_RF : 2D array, shape (R, F)
        Training data features assigned to right child using best split.
    y_R : 1D array, shape (R,)
        Training labels assigned to right child using best split.

    Examples
    --------
    # Example 1a: Simple example with F=1 and sorted features input
    >>> N = 6
    >>> F = 1
    >>> x_NF = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape((6, 1))
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id
    0
    >>> thresh_val
    2.5

    # Example 1b: Same as 1a but just scramble the order of x
    # Should give same results as 1a
    >>> x_NF = np.asarray([2.0, 1.0, 0.0, 3.0, 5.0, 4.0]).reshape((6, 1))
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id
    0
    >>> thresh_val
    2.5

    # Example 2: Advanced example with F=12 total features
    # Fill the features such that middle column is same as 1a above,
    # but the first 6 columns with random features
    # and the last 6 columns with all zeros
    >>> N = 6
    >>> F = 13
    >>> prng = np.random.RandomState(0)
    >>> x_N1 = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape((6,1))
    >>> x_NF = np.hstack([prng.randn(N, F//2), x_N1, np.zeros((N, F//2))])
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id
    6
    >>> thresh_val
    2.5

    # Example 3: binary split isn't possible (because all x same)
    >>> N = 5
    >>> F = 1
    >>> x_NF = np.asarray([3.0, 3.0, 3.0, 3.0, 3.0]).reshape((5,1))
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id is None
    True

    # Example 4: binary split isn't possible (because all y same)
    >>> N = 5
    >>> F = 3
    >>> prng = np.random.RandomState(0)
    >>> x_NF = prng.rand(N, F)
    >>> y_N  = 1.2345 * np.ones(N)
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id is None
    True
    '''
    N, F = x_NF.shape
    best_gini_gain = 0.0
    best_feat_id = None
    best_thresh_val = None
    x_LF, y_L, x_RF, y_R = None, None, None, None

    for f in range(F):
        # Sort the feature values and corresponding labels
        sorted_indices = np.argsort(x_NF[:, f])
        x_sorted = x_NF[sorted_indices, f]
        y_sorted = y_N[sorted_indices]

        for i in range(1, N):
            # Check if there is a valid split with at least MIN_SAMPLES_LEAF on each side
            if i < MIN_SAMPLES_LEAF or (N - i) < MIN_SAMPLES_LEAF:
                continue

            # Calculate Gini impurity for the left and right subsets
            gini_L = calculate_gini_impurity(y_sorted[:i])
            gini_R = calculate_gini_impurity(y_sorted[i:])

            # Calculate the weighted average of the Gini impurities
            weighted_gini = (i / N) * gini_L + ((N - i) / N) * gini_R

            # Calculate Gini gain
            gini_gain = calculate_gini_impurity(y_N) - weighted_gini

            # Update best split if the current split provides higher Gini gain
            if gini_gain > best_gini_gain:
                best_gini_gain = gini_gain
                best_feat_id = f
                best_thresh_val = (x_sorted[i - 1] + x_sorted[i]) / 2.0
                x_LF = x_NF[x_NF[:, f] <= best_thresh_val]
                y_L = y_N[x_NF[:, f] <= best_thresh_val]
                x_RF = x_NF[x_NF[:, f] > best_thresh_val]
                y_R = y_N[x_NF[:, f] > best_thresh_val]
    return (best_feat_id, best_thresh_val, x_LF, y_L, x_RF, y_R)



if __name__ == '__main__':
    # # Example 1a: Simple example with F=1 and sorted features input
    # N = 6
    # F = 1
    # x_NF = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape((6, 1))
    # y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    # feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    # print(feat_id)
    # print(thresh_val)
    
    #Example 2
    N = 6
    F = 13
    prng = np.random.RandomState(0)
    x_N1 = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape((6,1))
    x_NF = np.hstack([prng.randn(N, F//2), x_N1, np.zeros((N, F//2))])
    y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    print(feat_id)
    print(thresh_val)


#     # Example 4: binary split isn't possible (because all y same)
#     N = 5
#     F = 3
#     prng = np.random.RandomState(0)
#     x_NF = prng.rand(N, F)
#     y_N  = 1.2345 * np.ones(N)
#     feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
#     feat_id is None    
