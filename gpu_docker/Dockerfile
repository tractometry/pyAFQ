ARG NVIDIAVERSION=12.0.1-devel-ubuntu20.04
FROM nvidia/cuda:${NVIDIAVERSION}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# upgrade
RUN apt-get update && apt-get install --assume-yes apt-transport-https \
	ca-certificates gnupg software-properties-common \
	gcc git wget curl numactl cmake

# Miniconda3
RUN curl -L "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
    -o "/tmp/Miniconda3.sh"
RUN bash /tmp/Miniconda3.sh -b -p /opt/anaconda
RUN rm -rf /tmp/Miniconda3.sh
RUN cd /opt && eval "$(/opt/anaconda/bin/conda shell.bash hook)"
ENV PATH /opt/anaconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/anaconda/lib:${LD_LIBRARY_PATH}

# python prereqs
RUN conda install -c conda-forge git
RUN pip install numpy>=2.0.0
RUN pip install scipy>=1.13.0 cython nibabel dipy tqdm fslpy
RUN pip install git+https://github.com/dipy/GPUStreamlines.git

# clone pyAFQ GPUStreamlines
RUN git clone https://github.com/tractometry/pyAFQ.git /opt/pyAFQ
RUN cd /opt/pyAFQ && git reset --hard ${COMMIT}
RUN cd /opt/pyAFQ && pip install -e .
RUN /opt/pyAFQ/bin/pyAFQ download

ENTRYPOINT ["/opt/pyAFQ/bin/pyAFQ"]
