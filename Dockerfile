FROM python:3.9.19

LABEL Name=lotus Version=0.0.1

WORKDIR /app

COPY . .

RUN apt-get update && apt-get -y install cmake protobuf-compiler

RUN pip install cmake

RUN pip install -r requirements.txt

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install git+https://github.com/UT-Austin-RPL/Lotus.git

