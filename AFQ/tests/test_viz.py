import pytest

try:
    from AFQ.viz.utils import Viz
except ImportError:
    Viz = False


@pytest.mark.skipif(not Viz, "Skipping viz tests, unable to import viz utils")
def test_viz_name_errors():
    Viz("fury")

    with pytest.raises(
        TypeError,
        match="Visualization backend contain"
        + " either 'plotly' or 'fury'. "
        + "It is currently set to plotli",
    ):
        Viz("plotli")
