import fmflow as fm


def test_version():
    """Make sure the version is valid."""
    assert fm.__version__ == '0.2.6'


def test_author():
    """Make sure the author is valid."""
    assert fm.__author__ == 'astropenguin'
