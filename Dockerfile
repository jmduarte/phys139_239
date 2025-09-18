FROM ghcr.io/ucsd-ets/scipy-ml-notebook:2025.2-stable
LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

RUN apt-get update && \
    apt-get install -y imagemagick && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER jovyan

RUN mamba install -c conda-forge uproot xrootd root

RUN pip install --no-cache-dir 'xgboost==3.0.5' 'spektral==1.3.1' 'gdown==5.2.0' 'mplhep==0.4.1' 'jetnet==0.2.5' 'tables==3.10.2' 'torch_geometric' && \
    fix-permissions /opt/conda && \
    fix-permissions /home/jovyan

RUN pip install --no-cache-dir --no-index pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
