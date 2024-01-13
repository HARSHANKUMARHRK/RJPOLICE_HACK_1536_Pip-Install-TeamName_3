

FROM python:3.11-bullseye


# Set the working directory to /app
WORKDIR /RJPOLICE_HACK_1536_Pip-Install-TeamName_3

# Copy the current directory contents into the container at /app 
ADD . /RJPOLICE_HACK_1536_Pip-Install-TeamName_3

RUN apt-get update
RUN apt-get install 'libgl1-mesa-dev' -y
# Install the dependencies
RUN pip3 install -r requirements.txt

CMD ["python3","app.py"]