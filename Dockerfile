FROM continuumio/miniconda3:23.10.0-1 as acanalysis

SHELL ["/bin/bash", "-c"]

# Update conda and create clean environment
RUN conda update -y conda && \
    conda create -y -n ac -c conda-forge python=3.10 gcc=12.3.0 pip && \
    conda clean -a

COPY . /acanalysis

# Install /acanalysis
WORKDIR /acanalysis
RUN source activate ac && \
    pip install . && \
    conda clean -a

ENTRYPOINT ["/bin/bash", "/acanalysis/entrypoint.sh"]
WORKDIR /

