FROM pytorch-ngc

WORKDIR /workspace

COPY . .

RUN pip install -r requirements.txt
CMD ["python","generate_model.py"]


