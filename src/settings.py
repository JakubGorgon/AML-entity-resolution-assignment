import os
from dataclasses import dataclass


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    # Paths
    db_path: str = os.getenv("ER_DB_PATH", "data/clients.db")
    model_path: str = os.getenv("ER_MODEL_PATH", "models/entity_resolution_model.pkl")
    lsh_index_path: str = os.getenv("ER_LSH_INDEX_PATH", "models/lsh_index.pkl")
    minhashes_path: str = os.getenv("ER_MINHASHES_PATH", "models/minhashes.pkl")

    # LSH parameters (must be consistent across preprocessing/batch/api)
    lsh_threshold: float = _get_float("ER_LSH_THRESHOLD", 0.7)
    lsh_num_perm: int = _get_int("ER_LSH_NUM_PERM", 128)

    # API behavior
    rebuild_lsh_on_startup: bool = _get_bool("ER_REBUILD_LSH_ON_STARTUP", False)


settings = Settings()
