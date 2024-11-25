FROM continuumio/miniconda3:latest AS miniconda-stage

FROM debian:latest
RUN apt-get update && apt-get install -y \
    wget \
    git \
    gcc \
    build-essential \
    vim \
    wget \
    curl \
    git \
    unzip \
    tmux \
    && apt-get clean

COPY --from=miniconda-stage /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:${PATH}"
WORKDIR /root

RUN echo "export HF_ENDPOINT=https://hf-mirror.com" >> /root/.bashrc
COPY condarc /root/.condarc
COPY Dockerfile /root/dockerfile/Dockerfile
COPY pip.conf /root/.pip/pip.conf

RUN conda init bash
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
