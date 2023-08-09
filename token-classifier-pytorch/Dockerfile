FROM python:latest
WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY generate_model.py generate_model.py
 
RUN python generate_model.py


