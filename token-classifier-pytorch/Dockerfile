FROM python:latest
WORKDIR /workspace

#RUN apk add --no-cache gcc musl-dev linux-headers
#RUN pip install jupyter torch
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY generate_model.py generate_model.py
RUN python generate_model.py


#ENV FLASK_APP=app.py
#ENV FLASK_RUN_HOST=0.0.0.0
#EXPOSE 5000
#CMD ["flask", "run"]