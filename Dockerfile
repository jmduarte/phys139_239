FROM ghcr.io/ucsd-ets/scipy-ml-notebook:2024.2-stable
LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

RUN apt-get update && \
    apt-get install -y imagemagick && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER jovyan

RUN mamba install -c conda-forge uproot xrootd root

RUN pip install --no-cache-dir 'xgboost==2.0.3' 'spektral==1.3.1' 'gdown==5.1.0' 'mplhep==0.3.43' 'torch_geometric' && \
    fix-permissions /opt/conda && \
    fix-permissions /home/jovyan

RUN pip install --no-cache-dir --no-index pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cu118.html && \
    pip install --no-cache-dir typing-extensions --upgrade

RUN pip install --no-cache-dir 'jetnet' 'tables==3.8.0'

# USER $NB_UID:$NB_GID
# RUN mkdir -p /tmp/nvvm && mkdir -p /tmp/nvvm/libdevice && cp /opt/conda/lib/libdevice.10.bc /tmp/nvvm/libdevice/
# ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/tmp"
# ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/conda/lib
# ENV PATH=${PATH}:/usr/local/nvidia/bin:/opt/conda/bin:/datasets/software/R2019a/sys/cuda/glnxa64/cuda/bin

# larcv2 build
# ADD build_larcv2.sh /home/jovyan/build_larcv2.sh
# RUN source build_larcv2.sh
