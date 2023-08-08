FROM python:latest
WORKDIR /workspace

#RUN apk add --no-cache gcc musl-dev linux-headers
#RUN pip install jupyter torch
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


#ENV FLASK_APP=app.py
#ENV FLASK_RUN_HOST=0.0.0.0
#EXPOSE 5000
COPY . .
CMD ["flask", "run"]