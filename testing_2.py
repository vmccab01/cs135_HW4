import numpy as np
import unittest
from tree_utils import InternalDecisionNode, LeafNode

class TestDecisionTreeNodes(unittest.TestCase):

    def test_internal_node_predict(self):
        # Create an internal node
        feat_id = 0
        thresh_val = 3.5
        left_child = LeafNode(x_NF=np.array([[2.0], [3.0]]), y_N=np.array([1.0, 2.0]))
        right_child = LeafNode(x_NF=np.array([[4.0], [5.0]]), y_N=np.array([3.0, 4.0]))
        internal_node = InternalDecisionNode(x_NF=np.array([[3.0], [4.0]]), y_N=np.array([2.0, 3.0]),
                                             feat_id=feat_id, thresh_val=thresh_val,
                                             left_child=left_child, right_child=right_child)

        # Test predictions for a feature array
        x_TF = np.array([[2.0], [3.5], [4.0]])
        yhat_T = internal_node.predict(x_TF)
        expected_yhat_T = np.array([1.5, 2.5, 3.5])
        # print()
        np.testing.assert_array_almost_equal(yhat_T, expected_yhat_T)

    def test_leaf_node_predict(self):
        # Create a leaf node
        leaf_node = LeafNode(x_NF=np.array([[2.0], [3.0], [4.0]]), y_N=np.array([1.0, 2.0, 3.0]))

        # Test predictions for a feature array
        x_TF = np.array([[2.0], [3.5], [4.0]])
        yhat_T = leaf_node.predict(x_TF)
        expected_yhat_T = np.array([2.0, 2.0, 2.0])  # Mean of y_N is 2.0
        np.testing.assert_array_almost_equal(yhat_T, expected_yhat_T)

    def test_internal_node_str(self):
        # Create an internal node with leaf children
        feat_id = 0
        thresh_val = 3.5
        left_child = LeafNode(x_NF=np.array([[2.0], [3.0]]), y_N=np.array([1.0, 2.0]))
        right_child = LeafNode(x_NF=np.array([[4.0], [5.0]]), y_N=np.array([3.0, 4.0]))
        internal_node = InternalDecisionNode(x_NF=np.array([[3.0], [4.0]]), y_N=np.array([2.0, 3.0]),
                                             feat_id=feat_id, thresh_val=thresh_val,
                                             left_child=left_child, right_child=right_child)

        # Test string representation
        expected_str = (
            "Decision: X[0] < 3.500?\n"
            "  Y: Leaf: predict y = 1.500\n"
            "  N: Leaf: predict y = 3.500"
        )
        print(str(internal_node))
        print(expected_str)
        self.assertEqual(str(internal_node), expected_str)

    def test_leaf_node_str(self):
        # Create a leaf node
        leaf_node = LeafNode(x_NF=np.array([[2.0], [3.0], [4.0]]), y_N=np.array([1.0, 2.0, 3.0]))

        # Test string representation
        expected_str = "Leaf: predict y = 2.000"
        self.assertEqual(str(leaf_node), expected_str)

if __name__ == '__main__':
    unittest.main()