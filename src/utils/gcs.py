import os
import tarfile

from google.cloud import storage
from loguru import logger
from datetime import datetime
from config.env import get_env

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
    Args:
        bucket_name (str): Name of the GCS bucket to upload to.
        local_path (str): Local path to the file or directory to upload.
        destination_path (str): Destination path in the GCS bucket.
        timestamp (bool): If True, prepend a timestamp to the destination filename.
        compress (bool): If True and local_path is a directory, compress it before uploading.
        cleanup (bool): If True, remove temporary files after upload.
    Raises:
        ValueError: If the GCS credentials path is not set in the environment variables.
        Exception: If the upload fails for any reason.
    """
    # Check for credentials
    env = get_env()
    if not env.CREDENTIALS_PATH:
        logger.error("GCS credentials path is not set in the environment variables.")
        raise ValueError("GCS credentials path is not set.")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = env.CREDENTIALS_PATH

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
