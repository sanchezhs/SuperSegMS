import os
import tarfile

from google.cloud import storage
from loguru import logger
from datetime import datetime

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcs_api.json"


# def upload_directory_to_bucket(bucket_name: str, source_dir: str, destination_prefix: str = "", timestamp: bool = False) -> None:
#     logger.info(f"Uploading directory '{source_dir}' to bucket '{bucket_name}' with prefix '{destination_prefix}'")

#     destination_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#     destination_prefix = f"{destination_prefix}/{destination_id}" if destination_prefix else destination_id

#     try:
#         client = storage.Client()
#         bucket = client.bucket(bucket_name)

#         for root, _, files in os.walk(source_dir):
#             for file in files:
#                 local_path = os.path.join(root, file)
#                 relative_path = os.path.relpath(local_path, source_dir)
#                 blob_path = os.path.join(destination_prefix, relative_path).replace("\\", "/")

#                 blob = bucket.blob(blob_path)
#                 blob.upload_from_filename(local_path)
#                 logger.info(f"✅ Uploaded {local_path} → gs://{bucket_name}/{blob_path}")
#     except Exception as e:
#         logger.error(f"Error uploading directory to bucket: {e}")
#         raise e


def upload_file_to_bucket(
    bucket_name: str,
    local_path: str,
    destination_path: str,
    timestamp: bool = True,
    compress: bool = True,
    cleanup: bool = True,
) -> None:
    """
    Upload a single file or directory to a GCS bucket.
    If `compress` is True and the source is a directory, compress it before uploading.
    If `cleanup` is True, temporary files are deleted after upload.
    """
    logger.info(
        f"Uploading '{local_path}' to bucket '{bucket_name}' at '{destination_path}'"
    )

    if not os.path.exists(local_path):
        logger.error(f"Path '{local_path}' does not exist.")
        return

    if timestamp:
        destination_path = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{destination_path}"
        )

    is_temp_file = False

    if compress and os.path.isdir(local_path):
        os.makedirs("tmp", exist_ok=True)
        compressed_path = f"tmp/{os.path.basename(local_path)}.tar.gz"
        compress_directory(local_path, compressed_path)
        local_path = compressed_path
        is_temp_file = True
    elif compress and not os.path.isdir(local_path):
        logger.warning(
            f"Compression is enabled but '{local_path}' is not a directory. Skipping compression."
        )

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_path)
        blob.upload_from_filename(local_path)
        logger.info(f"✅ Uploaded {local_path} → gs://{bucket_name}/{destination_path}")
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e
    finally:
        if cleanup and is_temp_file:
            try:
                os.remove(local_path)
                logger.info(f"Removed temporary file {local_path}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Could not remove temp file: {cleanup_error}")


def compress_directory(source_dir: str, output_path: str):
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
