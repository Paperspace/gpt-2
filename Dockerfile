FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/
RUN python setup.py install