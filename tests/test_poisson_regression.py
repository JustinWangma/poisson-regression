from src.poisson_regression import generate_mock_data, fit_models


def test_poisson_fit_small():
    df = generate_mock_data(n=200, seed=123)
    results = fit_models(df)
    assert "linear" in results and "lq" in results
    assert results["linear"].x.shape[0] >= 4
