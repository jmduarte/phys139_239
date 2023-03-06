#!/usr/bin/env bash

function main() {
    source /opt/conda/etc/profile.d/conda.sh
    conda init bash
    conda activate base

    cd /home/jovyan/

    git clone https://github.com/DeepLearnPhysics/larcv2

    cd larcv2

    printf "\n# source configure.sh\n"
    source configure.sh

    printf "\n# LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}\n"
    printf "\n# PATH: ${PATH}\n"
    printf "\n# PYTHONPATH: ${PYTHONPATH}\n"

    printf "\n# make -j4\n"
    make -j4

}

main "$@" || exit 1
