ARG VARIANT=3.12-bookworm
FROM mcr.microsoft.com/devcontainers/python:${VARIANT}

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        libdbus-1-3 \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libnss3 \
        libopengl0 \
        libsm6 \
        libx11-xcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libwayland-client0 \
        libwayland-cursor0 \
        libwayland-egl1 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libxrandr2 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        xauth \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
