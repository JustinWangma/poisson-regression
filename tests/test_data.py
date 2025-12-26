from src.data import load_csv


def test_load_csv():
    df = load_csv("data/train.csv")
    assert "target" in df.columns
    assert len(df) >= 3
