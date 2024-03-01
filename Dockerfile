# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

# FROM nvidia/cuda:11.0.3-base-ubuntu20.04
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13.py310:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

WORKDIR /src

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    # --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.

COPY requirements.txt /src
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir -r /src/requirements.txt

RUN mkdir -p /src/images
RUN chown appuser /src/images

RUN mkdir -p /src/flagged
RUN chown appuser /src/flagged

RUN mkdir -p /src/cache_dir
RUN chown appuser /src/cache_dir

RUN mkdir -p /home/appuser
RUN chown appuser /home/appuser

# Copy the source code into the container.
COPY src/models /src/models
COPY src/main.py /src

# Expose the port that the application listens on.
EXPOSE 8080

# Switch to the non-privileged user to run the application.
USER appuser

# Run the application.
CMD python main.py
