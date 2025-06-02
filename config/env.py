import os
from schemas.pipeline_schemas import EnvConfig
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

def get_env() -> EnvConfig:
    return EnvConfig(
        CREDENTIALS_PATH=os.getenv("CREDENTIALS_PATH", "credentials.json"),
        CMB_API_KEY=os.getenv("CMB_API_KEY"),
        PHONE_NUMBER=os.getenv("PHONE_NUMBER"),
)
