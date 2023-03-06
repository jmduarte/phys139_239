FROM ucsdets/scipy-ml-notebook:2023.1-stable
LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

RUN apt-get update && \
    apt-get install -y imagemagick && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER jovyan

# RUN rm -rf /opt/conda/pkgs/cache
RUN mamba install -c conda-forge uproot xrootd root

RUN pip install --no-cache-dir 'xgboost==1.7.3' 'scikit-learn==1.2.1' 'spektral==1.2.0' 'gdown==4.6.0' 'mplhep==0.3.26' 'larcv==3.5.0' && \
    fix-permissions /opt/conda && \
    fix-permissions /home/jovyan

RUN pip install --no-cache-dir --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.0+cu111.html && \
    pip install --no-cache-dir torch-geometric && \
    pip install --no-cache-dir typing-extensions --upgrade

# RUN pip install --no-cache-dir 'jetnet==0.2.2'

RUN git clone https://github.com/DeepLearnPhysics/larcv2.git && cd larcv2 && source configure.sh && make

USER $NB_UID:$NB_GID
RUN mkdir -p /tmp/nvvm && mkdir -p /tmp/nvvm/libdevice && cp /opt/conda/lib/libdevice.10.bc /tmp/nvvm/libdevice/
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/tmp"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib
ENV PATH=${PATH}:/usr/local/nvidia/bin:/opt/conda/bin:/datasets/software/R2019a/sys/cuda/glnxa64/cuda/bin
