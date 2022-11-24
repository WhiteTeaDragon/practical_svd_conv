FROM nvcr.io/nvidia/pytorch:22.11-py3
RUN apt update
RUN apt install -y git
COPY requirements_reproducibility.txt requirements_reproducibility.txt
RUN pip install -r requirements_reproducibility.txt