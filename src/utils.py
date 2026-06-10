"""src/utils.py — 路径常量、配置加载、日志初始化。"""
from __future__ import annotations
import logging
import pathlib
from functools import lru_cache

import yaml

# ── 项目根目录（utils.py 所在的 src/ 的上一级）─────────────────
_SRC  = pathlib.Path(__file__).parent          # .../src
ROOT  = _SRC.parent                            # .../project_optimized

# ── 数据 / 模型 / 结果目录（自动创建）──────────────────────────
RAW     = ROOT / "data" / "raw";     RAW.mkdir(parents=True, exist_ok=True)
MODELS  = ROOT / "models";           MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results";          RESULTS.mkdir(parents=True, exist_ok=True)
LOGS    = ROOT / "logs";             LOGS.mkdir(parents=True, exist_ok=True)

# ── 配置文件路径 ─────────────────────────────────────────────────
_CONFIG_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    """
    加载 YAML 配置，带 lru_cache 避免重复 IO。
    path 为 None 时使用项目根目录的 config.yaml。
    """
    cfg_path = pathlib.Path(path) if path else _CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found: {cfg_path}\n"
            f"请确认 config.yaml 在项目根目录 {ROOT}"
        )
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 万2.5 保护性校验：commission_rate 不应超过 0.003
    comm = cfg.get("backtest", {}).get("commission_rate", None)
    if comm is not None and comm > 0.003:
        import warnings
        warnings.warn(
            f"commission_rate={comm} 超过 0.003，请检查是否填写了百分比而非小数！"
            f"（万2.5 应填 0.00025）",
            UserWarning, stacklevel=2
        )
    return cfg


def reload_config(path: str | None = None) -> dict:
    """清除缓存并重新加载配置（用于测试或热重载）。"""
    load_config.cache_clear()
    return load_config(path)


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """初始化日志：同时输出到控制台和可选文件。"""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        lp = LOGS / log_file
        handlers.append(logging.FileHandler(lp, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    logging.getLogger("baostock").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
