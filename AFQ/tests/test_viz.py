import pytest

try:
    import fury  # noqa: F401

    from AFQ.viz.utils import Viz

    test_viz = True
except ImportError:
    test_viz = False


@pytest.mark.skipif(
    not test_viz, reason="Skipping viz tests, unable to import viz utils"
)
def test_viz_name_errors():
    Viz("fury")

    with pytest.raises(
        TypeError,
        match="Visualization backend contain"
        + " either 'plotly' or 'fury'. "
        + "It is currently set to plotli",
    ):
        Viz("plotli")
