FROM ucsdets/scipy-ml-notebook:2022.3-stable
LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

RUN apt-get update && \
    apt-get install -y imagemagick && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER $NB_USER

RUN pip install --no-cache-dir xgboost
