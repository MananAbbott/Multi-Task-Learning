#!/usr/bin/env bash
set -euo pipefail

# Configuration
IMAGE_NAME="mtl-apprentice:latest"
BUILD_CONTEXT="."
HOST_PORT=8000
CONTAINER_PORT=8000

# Build the image
echo " Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" "$BUILD_CONTEXT"

# Run the container
echo "Running container on host port $HOST_PORT -> container port $CONTAINER_PORT"

# Uncomment the following line to run the container in the foreground
# docker run --rm -p "$HOST_PORT:$CONTAINER_PORT" "$IMAGE_NAME"


CID=$(docker run -d -p "$HOST_PORT:$CONTAINER_PORT" "$IMAGE_NAME")

echo "Waiting for server to be ready..."
sleep 5

echo
echo "Example #1: Breakthrough in AI research promises smarter chatbots."
curl -s -X POST http://localhost:$HOST_PORT/predict \
     -H "Content-Type: application/json" \
     -d '{"sentence":"Breakthrough in AI research promises smarter chatbots."}' \
     || echo "Request failed"
echo

echo "Example #2:Local team loses championship after a weak overtime performance."
curl -s -X POST http://localhost:$HOST_PORT/predict \
     -H "Content-Type: application/json" \
     -d '{"sentence":"Local team loses championship after a weak overtime performance."}' \
     || echo "Request failed"
echo

sleep 5

echo "Stopping container"
docker stop "$CID" >/dev/null