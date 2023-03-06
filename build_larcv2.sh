#!/usr/bin/env bash

function setNumProcessors () {
    # Set the number of processors used for build
    # to be 1 less than are available
    if [[ -f "$(which nproc)" ]]; then
        NPROC="$(nproc)"
    else
        NPROC="$(grep -c '^processor' /proc/cpuinfo)"
    fi
    echo `expr "${NPROC}" - 1`
}

function main() {
    cd /home/jovyan/

    git clone https://github.com/DeepLearnPhysics/larcv2

    cd larcv2

    printf "\n# source configure.sh\n"
    source configure.sh

    printf "\n# LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}\n"
    printf "\n# PATH: ${PATH}\n"
    printf "\n# PYTHONPATH: ${PYTHONPATH}\n"

    printf "\n# make -j${NPROC}\n"
    make -j${NPROC}

}

main "$@" || exit 1
