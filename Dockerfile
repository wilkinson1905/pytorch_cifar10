FROM nvcr.io/nvidia/pytorch:21.10-py3
RUN conda install -y mlflow
RUN conda install -y -c conda-forge gitpython
