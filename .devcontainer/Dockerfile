FROM continuumio/miniconda3
WORKDIR /workspace
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y git curl wget && rm -rf /var/lib/apt/lists/*
RUN conda init bash
CMD ["/bin/bash"]
