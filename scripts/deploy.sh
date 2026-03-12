#!/usr/bin/env bash
# Deploy SmokeSignal API to Azure Container Instances.
# Prerequisites: az CLI logged in, ACR created.
set -euo pipefail

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-smokesignal-rg}"
ACR_NAME="${AZURE_ACR_NAME:-smokesignalacr}"
IMAGE="$ACR_NAME.azurecr.io/smokesignal:latest"
CONTAINER_NAME="smokesignal-api"
CPU=1
MEMORY=1.5
PORT=8000

echo "==> Building image in ACR..."
az acr build --registry "$ACR_NAME" --image smokesignal:latest .

echo "==> Deploying to ACI..."
az container create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$CONTAINER_NAME" \
  --image "$IMAGE" \
  --cpu "$CPU" --memory "$MEMORY" \
  --ports "$PORT" \
  --dns-name-label smokesignal \
  --registry-login-server "$ACR_NAME.azurecr.io" \
  --registry-username "$(az acr credential show -n "$ACR_NAME" --query username -o tsv)" \
  --registry-password "$(az acr credential show -n "$ACR_NAME" --query 'passwords[0].value' -o tsv)"

FQDN=$(az container show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$CONTAINER_NAME" \
  --query ipAddress.fqdn -o tsv)

echo "==> Deployed! Endpoint: http://$FQDN:$PORT/score"
