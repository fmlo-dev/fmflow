import fmflow as fm


def test_version():
    """Make sure the version is valid."""
    assert fm.__version__ == '0.2.7'


def test_author():
    """Make sure the author is valid."""
    assert fm.__author__ == 'Akio Taniguchi'
