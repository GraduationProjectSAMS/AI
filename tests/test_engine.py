import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class RecommendationEngineTests(unittest.TestCase):

    def setUp(self):
        # Common mock data setup
        self.mock_users = pd.DataFrame([{
            "User ID": 123,
            "Purchase History": "[1]"
        }])
        
        self.mock_inventory = pd.DataFrame([{
            "ID": 1,
            "Room Type": "Living",
            "Aesthetic": "Modern",
            "Category": "Sofa",
            "Price": 250.0,
            "Color": "Black"
        }, {
            "ID": 2,
            "Room Type": "Living",
            "Aesthetic": "Modern",
            "Category": "Chair",
            "Price": 150.0,
            "Color": "Black"
        }])

        self.mock_order_items = pd.DataFrame([{
            "Order ID": 101,
            "Product ID": 1,
            "Unit Price": 250.0
        }])

    @patch('recommender.ai.engine.pd.read_excel')
    def test_get_recommendations_success(self, mock_read_excel):
        # Configure mock return values
        mock_read_excel.side_effect = [
            self.mock_inventory,  # Pieces
            self.mock_order_items,  # Order Items
            self.mock_users         # Users
        ]

        from recommender.ai.engine import get_recommendations_for_user
        result = get_recommendations_for_user(123)

        # Verify
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)  # Should recommend product 2
        self.assertEqual(result[0]["product_id"], 2)
        self.assertTrue(0 <= result[0]["score"] <= 1)

    @patch('recommender.ai.engine.pd.read_excel')
    def test_user_not_found(self, mock_read_excel):
        # Return empty user DataFrame with proper columns
        mock_read_excel.side_effect = [
            self.mock_inventory,
            self.mock_order_items,
            pd.DataFrame(columns=["User ID", "Purchase History"])  # Empty but with correct columns
        ]

        from recommender.ai.engine import get_recommendations_for_user
        result = get_recommendations_for_user(999)
        self.assertEqual(result, [])

    @patch('recommender.ai.engine.pd.read_excel')
    def test_empty_purchase_history(self, mock_read_excel):
        # User with empty purchase history
        empty_history_users = pd.DataFrame([{
            "User ID": 123,
            "Purchase History": "[]"
        }])
        
        mock_read_excel.side_effect = [
            self.mock_inventory,
            self.mock_order_items,
            empty_history_users
        ]

        from recommender.ai.engine import get_recommendations_for_user
        result = get_recommendations_for_user(123)
        self.assertEqual(result, [])

    @patch('recommender.ai.engine.pd.read_excel')
    def test_invalid_purchase_history_format(self, mock_read_excel):
        # User with malformed purchase history that will fail eval()
        invalid_history_users = pd.DataFrame([{
            "User ID": 123,
            "Purchase History": "[1, 2"  # Missing closing bracket
        }])
        
        mock_read_excel.side_effect = [
            self.mock_inventory,
            self.mock_order_items,
            invalid_history_users
        ]

        from recommender.ai.engine import get_recommendations_for_user
        with self.assertRaises(SyntaxError):
            get_recommendations_for_user(123)

    @patch('recommender.ai.engine.pd.read_excel')
    def test_no_matching_inventory(self, mock_read_excel):
        # User's purchases don't match any inventory items
        mock_read_excel.side_effect = [
            self.mock_inventory,
            self.mock_order_items,
            pd.DataFrame([{
                "User ID": 123,
                "Purchase History": "[99]"  # Non-existent product
            }])
        ]

        from recommender.ai.engine import get_recommendations_for_user
        result = get_recommendations_for_user(123)
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()