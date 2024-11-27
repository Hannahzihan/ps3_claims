import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)
    
    winsorizer = Winsorizer(lower_quantile=lower_quantile, upper_quantile=upper_quantile)

    winsorizer.fit(X)
    X_transformed = winsorizer.transform(X)

    lower_bound = winsorizer.lower_quantile_
    upper_bound = winsorizer.upper_quantile_

    assert np.all(X_transformed >= lower_bound), f"Values lower than lower bound {lower_bound} found"
    assert np.all(X_transformed <= upper_bound), f"Values higher than upper bound {upper_bound} found"

    if lower_quantile == 0.5 and upper_quantile == 0.5:
        assert np.allclose(X_transformed, np.median(X), atol=1e-5), "Not all values are clipped to the median"
