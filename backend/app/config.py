import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent

import tempfile

def _get_default_database_url() -> str:
    env_db = os.getenv("DATABASE_URL")
    if env_db:
        return env_db
    # In Vercel or serverless container, root fs is read-only
    if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        tmp_db = Path(tempfile.gettempdir()) / "orbital_twin.db"
        return f"sqlite:///{tmp_db.as_posix()}"
    try:
        test_file = BASE_DIR / ".write_test"
        test_file.touch(exist_ok=True)
        test_file.unlink(missing_ok=True)
        return f"sqlite:///{BASE_DIR}/orbital_twin.db"
    except (OSError, PermissionError):
        tmp_db = Path(tempfile.gettempdir()) / "orbital_twin.db"
        return f"sqlite:///{tmp_db.as_posix()}"

class Settings(BaseSettings):
    APP_NAME: str = "ORBITAL TWIN"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database
    DATABASE_URL: str = _get_default_database_url()
    
    # NASA SMAP/MSL Dataset path
    NASA_DATA_PATH: str = os.getenv(
        "NASA_DATA_PATH", 
        str(BASE_DIR / "spacecraft-digital-twin-data" / "01_nasa_smap_msl")
    )
    
    # Grok AI Configuration
    GROK_API_KEY: str = os.getenv("GROK_API_KEY", "")
    GROK_MODEL: str = os.getenv("GROK_MODEL", "grok-beta")
    GROK_BASE_URL: str = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
    
    # Models directory
    MODELS_DIR: str = os.getenv("MODELS_DIR", str(BASE_DIR / "models"))
    
    # Simulation settings
    SIMULATION_TICK_SECONDS: float = 1.0
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
