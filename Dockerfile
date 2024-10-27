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
    && apt-get clean

COPY --from=miniconda-stage /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:${PATH}"
WORKDIR /root

CMD ["/bin/bash"]
