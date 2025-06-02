import os
from schemas.pipeline_schemas import EnvConfig
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

def get_env() -> EnvConfig:
    return EnvConfig(
        GCS_BUCKET=os.getenv("GCS_BUCKET", "tfm-training-results"),
        GCS_DEST_PATH=os.getenv("GCS_DEST_PATH", "resultados.tar.gz"),
        CMB_API_KEY=os.getenv("CMB_API_KEY"),
        PHONE_NUMBER=os.getenv("PHONE_NUMBER"),
)
