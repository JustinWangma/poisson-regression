from pathlib import Path
from src.train import train


def test_train_saves_model(tmp_path):
    out_dir = tmp_path / "models_test"
    train(data_path="data/train.csv", out_dir=str(out_dir))
    assert (out_dir / "model.joblib").exists()
