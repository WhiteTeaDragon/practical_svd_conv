FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN apt update
RUN apt install -y git
COPY requirements_reproducibility.txt requirements_reproducibility.txt
RUN pip install -r requirements_reproducibility.txt
