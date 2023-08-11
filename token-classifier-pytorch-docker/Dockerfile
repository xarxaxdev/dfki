FROM docker.io/gnuoctave/octave:8.2.0

WORKDIR /workspace

COPY . .

RUN pip install -r requirements.txt
CMD ["python","generate_model.py"]


