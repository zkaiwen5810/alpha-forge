# MinIO Essentials

MinIO is a local S3-compatible object store. In this repo, LiteLLM uses it as cold storage for Responses API session continuity.

## Services

- `minio`: the actual storage server. It exposes:
  - S3 API on `http://localhost:9000`
  - web console on `http://localhost:9001`
- `minio-init`: a one-time setup job. It waits for `minio`, logs in with the configured credentials, creates the bucket, and exits.

## Why there are two services

- `minio` provides storage.
- `minio-init` removes manual setup by creating the bucket automatically.

## How the flow works

1. `minio` starts.
2. `minio-init` connects to `http://minio:9000`.
3. `minio-init` creates `MINIO_BUCKET_NAME` if it does not already exist.
4. `litellm` writes cold-storage objects into that bucket.

## Key config

- `.devcontainer/minio.env`
  - `MINIO_ROOT_USER`: admin username
  - `MINIO_ROOT_PASSWORD`: admin password
  - `MINIO_BUCKET_NAME`: bucket LiteLLM uses
  - `MINIO_REGION`: S3-compatible region metadata
- `.devcontainer/litellm_config.yaml`
  - `s3_v2`: tells LiteLLM to use an S3-compatible backend
  - `s3_endpoint_url: http://minio:9000`: points LiteLLM at local MinIO
  - `store_prompts_in_cold_storage: true`: enables persistence for session continuity

## Mental model

- `minio` = local object storage server
- `minio-init` = startup script that creates the bucket LiteLLM needs
