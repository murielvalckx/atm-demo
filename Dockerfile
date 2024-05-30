###########
# BUILDER #
###########

# pull official base image
#FROM python:3.10.7-slim-buster as builder
FROM python:3.12-rc-slim as builder

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Update pip
RUN pip install --upgrade pip

# install dependencies
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

#########
# FINAL #
#########
FROM python:3.12-rc-slim

# create directory for the app user
RUN mkdir -p /data/atm

# create the app user
RUN addgroup --system flaskuser && adduser --system --group flaskuser

# create the appropriate directories
ENV HOME=/data/atm
ENV APP_HOME=/data/atm/web
#RUN mkdir $APP_HOME
WORKDIR $APP_HOME

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends netcat-traditional
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache /wheels/*

# copy project
#COPY . $APP_HOME

# Ser ownership
RUN chown -R flaskuser:flaskuser $APP_HOME

# Change user
USER flaskuser
