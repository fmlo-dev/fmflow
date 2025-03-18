# standard library
from sys import argv
from pathlib import Path


# dependencies
from fmflow.fits.nro45m.functions import read_backendlog_sam45


# constants
HELP = "$ python dump.py /path/to/log"


if __name__ == "__main__":
    if len(argv) != 2:
        raise RuntimeError(HELP)

    log = Path(argv[1]).resolve()
    hdu = read_backendlog_sam45(log, "<")
    hdu.writeto(log.with_suffix(".fits"))
