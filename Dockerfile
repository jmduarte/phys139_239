FROM ucsdets/scipy-ml-notebook:2023.1-stable
LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

RUN apt-get update && \
    apt-get install -y imagemagick && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER jovyan

RUN mamba install -c conda-forge uproot xrootd root

RUN pip install --no-cache-dir 'xgboost==1.7.3' 'scikit-learn==1.2.1' 'spektral==1.2.0' 'gdown==4.6.0' 'mplhep==0.3.26' && \
    fix-permissions /opt/conda && \
    fix-permissions /home/jovyan

RUN pip install --no-cache-dir --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.0+cu111.html && \
    pip install --no-cache-dir torch-geometric && \
    pip install --no-cache-dir typing-extensions --upgrade

# RUN pip install --no-cache-dir 'jetnet==0.2.2'

USER $NB_UID:$NB_GID
RUN mkdir -p /tmp/nvvm && mkdir -p /tmp/nvvm/libdevice && cp /opt/conda/lib/libdevice.10.bc /tmp/nvvm/libdevice/
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/tmp"
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/conda/lib
ENV PATH=${PATH}:/usr/local/nvidia/bin:/opt/conda/bin:/datasets/software/R2019a/sys/cuda/glnxa64/cuda/bin

# larcv build
# ENV LARCV_BASEDIR=/home/jovyan/larcv2
# ENV LARCV_BUILDDIR="${LARCV_BASEDIR}/build"
# ENV LARCV_COREDIR="${LARCV_BASEDIR}/larcv/core"
# ENV LARCV_APPDIR="${LARCV_BASEDIR}/larcv/app"
# ENV LARCV_LIBDIR="${LARCV_BUILDDIR}/lib"
# ENV LARCV_INCDIR="${LARCV_BUILDDIR}/include"
# ENV LARCV_BINDIR="${LARCV_BUILDDIR}/bin"
# ENV LARCV_ROOT6=1
# ENV LARCV_CXX=g++
# # with numpy
# ENV LARCV_NUMPY=1
# ENV LARCV_INCLUDES="-I/home/jovyan/larcv2/build/include -I/opt/conda/include/python3.9 -I/opt/conda/include/python3.9 -I/opt/conda/lib/python3.9/site-packages/numpy/core/include"
# ENV LARCV_LIBS="-L/opt/conda/lib/ -L/opt/conda/lib/python3.9/config-3.9-x86_64-linux-gnu -L/opt/conda/lib -lcrypt -lpthread -ldl -lutil -lrt -lm -lm -L/home/jovyan/larcv2/build/lib -llarcv"
# ENV LARCV_PYTHON=/opt/conda/bin/python3
# ENV LARCV_PYTHON_CONFIG=python3.9-config
# # set bin and lib path
# ENV PATH=${LARCV_BASEDIR}/bin:${LARCV_BINDIR}:${PATH}
# ENV LD_LIBRARY_PATH=${LARCV_LIBDIR}:${LD_LIBRARY_PATH}
# ENV PYTHONPATH=${LARCV_BASEDIR}/python:${PYTHONPATH}
# # build larcv
# RUN cd /home/jovyan && \
#     git clone https://github.com/DeepLearnPhysics/larcv2 && \
#     cd larcv2 && \
#     mkdir -p $LARCV_BUILDDIR && \
#     mkdir -p $LARCV_LIBDIR && \
#     mkdir -p $LARCV_BINDIR && \
#     make -j4
ADD build_larcv2.sh /home/jovyan/build_larcv2.sh
RUN source build_larcv2.sh
