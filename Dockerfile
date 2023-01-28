FROM ucsdets/scipy-ml-notebook:2023.1-stable
LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

RUN apt-get update && \
    apt-get install -y imagemagick && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER $NB_USER

RUN pip install -U --no-cache-dir xgboost scikit-learn
