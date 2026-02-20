from pathlib import Path
from unittest.mock import patch
from app.src.loader import InMemoryModelStorage


@patch("app.src.main.pickle")
def test_prediction(mock_pickle, client):

    model_path = Path(__file__).resolve().parent / "models" / "fake_model.txt"

    with patch("app.src.main.storage", InMemoryModelStorage()):
        with model_path.open("rb") as model_file:
            upload_response = client.post(
                "/model/upload", files={"file": ("model.pkl", model_file)}
            )

            key = upload_response.json()["key"]

        response = client.post(
            "/model/predict", json={"id": key, "features": [0.1, 0.2, 0.07, 0.14]}
        )
        expected_response = {
            "storage_used": "InMemoryModelStorage",
            "data": {"model_answer": 1},
        }

        assert response.json() == expected_response
