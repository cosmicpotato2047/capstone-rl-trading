"""experiment_config.yaml 로더."""
import yaml
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "experiment_config.yaml"


def load_config(path: str | Path = _DEFAULT_CONFIG) -> dict:
    """YAML 설정 파일을 dict로 반환한다."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
