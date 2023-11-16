from tree_utils import InternalDecisionNode, LeafNode
import numpy as np
import unittest

class TestInternalDecisionNode(unittest.TestCase):
    def test_predict_with_leaf_children(self):
        # Test when both left and right children are leaf nodes
        x_NF = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_N = np.array([0.0, 1.0, 1.0])
        feat_id = 0
        thresh_val = 3.0

        left_child = LeafNode(x_NF[:2, :], y_N[:2])
        right_child = LeafNode(x_NF[2:, :], y_N[2:])

        node = InternalDecisionNode(x_NF, y_N, feat_id, thresh_val, left_child, right_child)

        # Test prediction for input array
        x_TF = np.array([[2.0, 3.0], [4.0, 5.0]])
        yhat_T = node.predict(x_TF)

        # Expected predictions based on the threshold
        expected_yhat_T = np.array([0.5, 1.0])
        np.testing.assert_allclose(yhat_T, expected_yhat_T, atol=1e-3)

    def test_predict_with_internal_children(self):
        # Test when both left and right children are internal decision nodes
        x_NF = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_N = np.array([0.0, 1.0, 1.0])
        feat_id = 0
        thresh_val = 3.0

        left_child = LeafNode(x_NF[:2, :], y_N[:2])
        right_child = InternalDecisionNode(x_NF[2:, :], y_N[2:], feat_id=1, thresh_val=5.0, left_child=None, right_child=None)

        node = InternalDecisionNode(x_NF, y_N, feat_id, thresh_val, left_child, right_child)

        # Test prediction for input array
        x_TF = np.array([[2.0, 3.0], [4.0, 5.0]])
        yhat_T = node.predict(x_TF)

        # Expected predictions based on the threshold and feature id
        expected_yhat_T = np.array([0.5, 1.0])
        np.testing.assert_allclose(yhat_T, expected_yhat_T, atol=1e-3)

    def test_str_representation(self):
        # Test string representation of the internal decision node
        x_NF = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_N = np.array([0.0, 1.0, 1.0])
        feat_id = 0
        thresh_val = 3.0

        left_child = LeafNode(x_NF[:2, :], y_N[:2])
        right_child = LeafNode(x_NF[2:, :], y_N[2:])

        node = InternalDecisionNode(x_NF, y_N, feat_id, thresh_val, left_child, right_child)

        expected_str = (
            "Decision: X[0] < 3.000?\n"
            "  Y: Leaf: predict y = 0.000\n"
            "  N: Leaf: predict y = 1.000"
        )

        self.assertEqual(str(node), expected_str)

if __name__ == '__main__':
    unittest.main()