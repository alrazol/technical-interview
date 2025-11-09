from pydantic_settings import BaseSettings
import dotenv
from pathlib import Path
from dotenv import find_dotenv

dotenv.load_dotenv(find_dotenv(".env"))


class Settings(BaseSettings):
    cache_dir: Path
    artifacts_dir: Path
