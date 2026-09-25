#!/bin/sh
# Build for linux/amd64 and push to the EPFL registry - run from the infra/ directory

set -eu

if [ -f .env ]; then
    set -a; . ./.env; set +a
else
    echo "Error: .env file not found."
    exit 1
fi

echo "IMAGE_NAME: $IMAGE"

TARGET_PLATFORM="linux/amd64"
IMAGE_NAME=$IMAGE

docker buildx build \
    -f Dockerfile \
    --platform "$TARGET_PLATFORM" \
    -t "$IMAGE_NAME" \
    --load .

# run `docker login registry.rcp.epfl.ch` if not logged in
docker push "$IMAGE_NAME"

echo "Pushed $IMAGE_NAME"