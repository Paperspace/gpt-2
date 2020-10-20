FROM tiangolo/uvicorn-gunicorn-fastapi

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/