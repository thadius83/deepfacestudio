# syntax=docker/dockerfile:1.4  # enable BuildKit features for caching
FROM tensorflow/tensorflow:latest-gpu

# build mode switch for dev vs prod
ARG BUILD_MODE=dev
ENV BUILD_MODE=${BUILD_MODE}

# speed up installs & prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# copy and install Python dependencies
COPY requirements.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip && \
    if [ "$BUILD_MODE" = "dev" ]; then \
        pip install --no-cache-dir --ignore-installed -r /tmp/requirements.txt; \
    else \
        pip install --no-cache-dir --ignore-installed -r /tmp/requirements.txt; \
    fi

# install system libraries for OpenCV runtime
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# copy application code
WORKDIR /srv
COPY app ./app

# make sure data mountpoint exists
RUN mkdir -p /data/reference_db

EXPOSE 3900
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3900"]
