#!/bin/bash

cat << EOS
INFO: this will build fmflow/testdocs
(this will not change any files of fmflow/docs)

EOS

if type sphinx-build >/dev/null 2>&1; then
    sphinx-apidoc -f -o ./apis ../../fmflow
    sphinx-build -a -d ../../testdocs/_doctree ./ ../../testdocs
    sphinx-build -a -d ../../testdocs/_doctree ./ ../../testdocs
else
    echo "ERROR: sphinx is not installed"
    exit 1
fi
