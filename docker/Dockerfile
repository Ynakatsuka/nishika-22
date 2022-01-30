ARG BASE_TAG=staging

FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04 AS nvidia
FROM gcr.io/kaggle-images/python:${BASE_TAG}

# Cuda support
COPY --from=nvidia /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/
COPY --from=nvidia /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/
COPY --from=nvidia /etc/apt/trusted.gpg /etc/apt/trusted.gpg.d/cuda.gpg
COPY --from=nvidia /usr/local/cuda/bin/nvcc /usr/local/bin/
COPY --from=nvidia /usr/local/cuda-11.3/targets/x86_64-linux/include/ /usr/local/include/
COPY --from=nvidia /usr/local/cuda-11.3/nvvm/bin/cicc /usr/local/bin/
COPY --from=nvidia /usr/local/cuda-11.3/bin/ /usr/local/bin/
COPY --from=nvidia /usr/local/cuda-11.3/targets/x86_64-linux/lib/ /usr/local/lib/

# See b/142337634#comment28
RUN sed -i 's/deb https:\/\/developer.download.nvidia.com/deb http:\/\/developer.download.nvidia.com/' /etc/apt/sources.list.d/*.list

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/bin:${PATH}

# reinstall pytorch
RUN conda remove --force -y pytorch torchvision torchaudio torchtext cpuonly
RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
WORKDIR /

# libturbojpeg
RUN apt-get install -y libturbojpeg

# requirements
ADD requirements.txt  /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
