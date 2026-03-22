FROM mcr.microsoft.com/devcontainers/python:3.12

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \ 
        rclone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*