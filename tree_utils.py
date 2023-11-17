"""
tree_utils.py

Defines two Python classes, one for each kind of nodes:
- InternalDecisionNode
- LeafNode

Your job is to edit the *predict* method for each node.

Examples
--------
>>> N = 6
>>> F = 1
>>> x_NF = np.linspace(-5, 5, N).reshape((N,F))
>>> y_N = np.hstack([np.linspace(0, 1, N//2), np.linspace(-1, 0, N//2)])

>>> feat_id = 0
>>> thresh_val = 0.0
>>> left_mask_N = x_NF[:, feat_id] < thresh_val
>>> right_mask_N = np.logical_not(left_mask_N)
>>> left_leaf = LeafNode(x_NF[left_mask_N], y_N[left_mask_N])
>>> right_leaf = LeafNode(x_NF[right_mask_N], y_N[right_mask_N])

>>> left_leaf.y_N
array([0. , 0.5, 1. ])

>>> root = InternalDecisionNode(
...     x_NF, y_N, feat_id, thresh_val, left_leaf, right_leaf)

# Display the tree
>>> print(root)
Decision: X[0] < 0.000?
  Y: Leaf: predict y = 0.500
  N: Leaf: predict y = -0.500

# Remember the true label of each node in train set
>>> y_N
array([ 0. ,  0.5,  1. , -1. , -0.5,  0. ])

# Predictions of the whole 3-node tree for each example in training set
>>> yhat_N = root.predict(x_NF)
>>> np.round(yhat_N, 4)
array([ 0.5,  0.5,  0.5, -0.5, -0.5, -0.5])

# Predictions of the left leaf for each example in training set:
>>> yhat_N = left_leaf.predict(x_NF)
>>> np.round(yhat_N, 4)
array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

# Predictions of the right leaf for each example in training set:
>>> yhat_N = right_leaf.predict(x_NF)
>>> np.round(yhat_N, 4)
array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5])

# Predictions for new input never seen before
>>> np.round(root.predict(x_NF[::-1] + 1.23), 4)
array([-0.5, -0.5, -0.5, -0.5,  0.5,  0.5])

"""

import numpy as np

class InternalDecisionNode(object):

    '''
    Defines a single node used to make yes/no decisions within a binary tree.

    Attributes
    ----------
    x_NF : 2D array, shape (N,F)
        Feature vectors of the N training examples that have reached this node.
    y_N : 1D array, shape (N,)
        Labels of the N training examples that have reached this node.
    feat_id : int
        Which feature this node will split on.
    thresh_val : float
        The value of the threshold used to divide input examples to either the
        left or the right child of this node.
    left_child : instance of InternalDecisionNode or LeafNode class
        Use to make predictions for examples less than this node's threshold.
    right_child : instance of InternalDecisionNode or LeafNode class
        Use to make predictions for examples greater than this node's threshold.
    '''

    def __init__(self, x_NF, y_N, feat_id, thresh_val, left_child, right_child):
        self.x_NF = x_NF
        self.y_N = y_N
        self.feat_id = feat_id
        self.thresh_val = thresh_val
        self.left_child = left_child
        self.right_child = right_child


    def predict(self, x_TF):
        ''' Make prediction given provided feature array

        For an internal node, we assign each input example to either our
        left or right child to get its prediction.
        We then aggregate the results into one array to return.

        Args
        ----
        x_TF : 2D numpy array, shape (T, F)

        Returns
        -------
        yhat_T : 1D numpy array, shape (T,)
        '''
        T, F = x_TF.shape


        # Left and Right masks
        # print(type(self.left_child))
        # print(type(self.right_child))
        left_mask_T = x_TF[:, self.feat_id] < self.thresh_val
        right_mask_T = x_TF[:, self.feat_id] >= self.thresh_val
        #right_mask_T = np.logical_not(left_mask_T)
        ## covering if one or both children does not exist

        if self.left_child is not None:
            # print('left child')
            left_kid_pred = self.left_child.predict(x_TF[left_mask_T, :])
            # print('left child done')
        else:
            left_kid_pred = np.zeros(np.sum(left_mask_T), dtype=np.float64)
        
        if self.right_child is not None:
            # print('right child')
            right_kid_pred = self.right_child.predict(x_TF[right_mask_T, :])
            # print('right child done')
        else:
            right_kid_pred = np.zeros(np.sum(right_mask_T), dtype=np.float64)

        if self.left_child is None and self.right_child is None:
            return LeafNode.predict(self, x_TF)
        ## Return yhat

        yhat_T = np.zeros(T, dtype=np.float64)
        yhat_T[left_mask_T] = left_kid_pred
        yhat_T[right_mask_T] = right_kid_pred


        # print('Left Mask:')
        # print(left_mask_T)
        # print('Right Mask:')
        # print(right_mask_T)
        # print('Left Kid Pred:')
        # print(left_kid_pred)
        # print('Right Kid Pred:')
        # print(right_kid_pred)
        # print('yhat_T')
        # print(yhat_T)
        return yhat_T


    def __str__(self):
        ''' Pretty print a string representation of this node
        
        Returns
        -------
        s : string
        '''
        left_str = self.left_child.__str__()
        right_str = self.right_child.__str__()
        lines = [
            "Decision: X[%d] < %.3f?" % (self.feat_id, self.thresh_val),
            "  Y: " + left_str.replace("\n", "\n    "),
            "  N: " + right_str.replace("\n", "\n    "),
            ]
        return '\n'.join(lines)


class LeafNode(object):
    
    '''
    Defines a single node within a binary tree that makes constant predictions.

    We assume the objective function is to minimize squared error on the train
    set. This means the optimal thing to do is to predict the mean of all the
    train examples that reach the region defined by this leaf.

    Attributes
    ----------
    x_NF : 2D array, shape (N,F)
        Feature vectors of the N training examples that have reached this node.
    y_N : 1D array, shape (N,)
        Labels of the N training examples that have reached this node.
        This may be a subset of all the training examples.
    '''

    def __init__(self, x_NF, y_N):
        self.x_NF = x_NF
        self.y_N = y_N


    def predict(self, x_TF):
        ''' Make prediction given provided feature array
        
        For a leaf node, all input examples get the same predicted value,
        which is determined by the mean of the training set y values
        that reach this node.

        Args
        ----
        x_TF : 2D numpy array, shape (T, F)

        Returns
        -------
        yhat_T : 1D numpy array, shape (T,)
            Predicted y value for each provided example
        '''
        T = x_TF.shape[0]
        
        yhat_T = np.mean(self.y_N) * np.ones(T)
        # print('y_N in leaf')
        # print(self.y_N)
        # print('yhat_T in leaf')
        # print(yhat_T) 
        return yhat_T


    def __str__(self):
        ''' Pretty print a string representation of this node
        
        Returns
        -------
        s : string
        '''        
        return "Leaf: predict y = %.3f" % np.mean(self.y_N)



if __name__ == '__main__':
    # Just does the same as doctest above, to help with debugging.

    N = 6
    F = 1
    x_NF = np.linspace(-5, 5, N).reshape((N,F))
    y_N = np.hstack([np.linspace(0, 1, N//2), np.linspace(-1, 0, N//2)])

    feat_id = 0
    thresh_val = 0.0
    left_mask_N = x_NF[:, feat_id] < thresh_val
    right_mask_N = np.logical_not(left_mask_N)
    left_leaf = LeafNode(x_NF[left_mask_N], y_N[left_mask_N])
    right_leaf = LeafNode(x_NF[right_mask_N], y_N[right_mask_N])
    root = InternalDecisionNode(
        x_NF, y_N, feat_id, thresh_val, left_leaf, right_leaf)

    print("Displaying the tree")
    print(root)

    print("Predictions of the whole 3-node tree for each example in training set:")
    yhat_N = root.predict(x_NF)
    print(np.round(yhat_N, 4))
    print("Predictions of the left leaf for each example in training set:")
    yhat_N = left_leaf.predict(x_NF)
    print(np.round(yhat_N, 4))
    print("Predictions of the right leaf for each example in training set:")
    yhat_N = right_leaf.predict(x_NF)
    print(np.round(yhat_N, 4))
    print("# Predictions for new input never seen before")
    print(np.round(root.predict(x_NF[::-1] + 1.23), 4))