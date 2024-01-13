FROM python:3.11-slim-buster

WORKDIR /RJPOLICE_HACK_1536_Pip-Install-TeamName_3

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
CMD ["python3","app.py"]