#!/bin/bash

prompt () {
    echo -ne "${1} (y/[n]) "
    read answer
    case ${answer} in
        [yY]* ) return 0;;
        *     ) return 1;;
    esac
}

cat << EOS
WARN: this will (re)write all files of fmflow/docs
we recommend to use testbuild.sh to build fmflow/testdocs
as a test build before you actually build fmflow/docs

EOS

if prompt "continue to build?"; then
    if type sphinx-build >/dev/null 2>&1; then
        sphinx-apidoc -f -o ./apis ../../fmflow
        sphinx-build -a -d ../_doctree ./ ../
        sphinx-build -a -d ../_doctree ./ ../
    else
        echo "ERROR: sphinx is not installed"
        exit 1
    fi
fi
