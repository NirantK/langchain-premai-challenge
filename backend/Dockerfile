# https://hub.docker.com/_/python
FROM python:3.9.16-slim-bullseye

ENV PYTHONUNBUFFERED True

RUN pip install --upgrade pip

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install wget and sudo
RUN apt-get update && \
    apt-get install -y wget sudo

# Download and execute the installation script
RUN wget -q https://get.prem.ninja/install.sh -O install.sh

RUN sudo bash ./install.sh

COPY . ./

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]