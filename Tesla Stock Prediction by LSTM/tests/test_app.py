"""
Integration tests for Flask application routes and API endpoints.
"""

import unittest
from app import app


class TestFlaskRoutes(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_homepage_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"StockAI", response.data)
        self.assertIn(b"AI-Powered Stock Market", response.data)

    def test_dashboard_loads_with_ticker(self):
        response = self.client.get("/dashboard?ticker=RELIANCE.NS")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"RELIANCE.NS", response.data)

    def test_api_stock_invalid_ticker(self):
        response = self.client.get("/api/stock/INVALID$$%TICKER")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Invalid ticker format", data["error"])

    def test_api_predict_untrained_model(self):
        response = self.client.post("/api/predict", json={
            "ticker": "NON_EXISTENT_STOCK_999",
            "days": 7
        })
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertTrue(data.get("needs_training"))

    def test_api_train_invalid_ticker(self):
        response = self.client.post("/api/train", json={
            "ticker": "INVALID$$%",
            "epochs": 5
        })
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])


if __name__ == "__main__":
    unittest.main()
