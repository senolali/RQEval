"""
LLM Reasoning Quality Evaluation Framework
===========================================
FULLY CONFIG-DRIVEN -- no code changes needed to add/remove:
  * Models    -> edit config.yaml [models] section
  * Datasets  -> edit config.yaml [datasets] section
  * Weights   -> edit config.yaml [aggregation] section

Run: python main.py
Run specific config: python main.py --config path/to/config.yaml
"""

"""
LLM Reasoning Quality Evaluation Framework
===========================================
FULLY CONFIG-DRIVEN -- no code changes needed to add/remove:
  * Models    -> edit config.yaml [models] section
  * Datasets  -> edit config.yaml [datasets] section
  * Weights   -> edit config.yaml [aggregation] section

Run: python main.py
Run specific config: python main.py --config path/to/config.yaml
"""

import argparse
import os
import sys
import time
import warnings
import yaml

warnings.filterwarnings("ignore", message=".*logits.*model output.*")


def _install_certifi_ssl_fallback() -> None:
    """Use Certifi only when the native Windows CA store is unreadable.

    Some Windows/Conda OpenSSL combinations raise ASN1: NOT_ENOUGH_DATA while
    loading the Windows certificate store.  Verification remains enabled:
    this fallback changes only the CA bundle, from the failing native store
    to Certifi's Mozilla CA bundle.
    """
    import ssl

    try:
        ssl.create_default_context()
        return
    except ssl.SSLError as exc:
        if "ASN1: NOT_ENOUGH_DATA" not in str(exc):
            raise

    try:
        import certifi
    except ImportError:
        warnings.warn(
            "The Windows certificate store is unreadable and Certifi is not "
            "installed. Install it with: python -m pip install certifi"
        )
        return

    original_create_default_context = ssl.create_default_context

    def certifi_default_context(
        purpose=ssl.Purpose.SERVER_AUTH,
        *,
        cafile=None,
        capath=None,
        cadata=None,
    ):
        if cafile is None and capath is None and cadata is None:
            cafile = certifi.where()
        return original_create_default_context(
            purpose=purpose,
            cafile=cafile,
            capath=capath,
            cadata=cadata,
        )

    ssl.create_default_context = certifi_default_context
    ssl._create_default_https_context = certifi_default_context
    warnings.warn(
        "Windows CA store could not be parsed; using Certifi's verified CA "
        "bundle for HTTPS connections."
    )


_install_certifi_ssl_fallback()

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# HuggingFace login (optional)
try:
    from huggingface_hub import login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)
except ImportError:
    pass

_base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base_dir)

from utils.logger import get_logger
from utils.reproducibility import set_seed, enable_deterministic_mode
import logging
logger = get_logger(__name__)

def _set_log_level(config: dict) -> None:
    """Set global log level from config. verbose: true -> DEBUG."""
    exp     = config.get("experiment", {})
    verbose = exp.get("verbose", False)
    level   = logging.DEBUG if verbose else logging.INFO

    # Set root logger
    logging.getLogger().setLevel(level)

    # Set all existing loggers AND their handlers
    for name in list(logging.root.manager.loggerDict):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        for handler in lg.handlers:
            handler.setLevel(level)

    # Also fix root logger handlers
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)


# -------------------------------------------------------------
# Registry: model type -> builder function
# Adding a new model type: add one entry here, create the class
# -------------------------------------------------------------

def _get_temperature(p, det):
    """Return effective temperature and deterministic flag from params."""
    temp = p.get("temperature", None)
    if temp is not None and float(temp) > 0:
        det = False   # explicit temperature overrides deterministic mode
    return temp, det


def _build_openai(name, p, mc, det):
    from models.openai_model import OpenAIModel
    key = os.environ.get(p.get("api_key_env", ""), "")
    if not key:
        raise EnvironmentError(f"Env var '{p.get('api_key_env')}' not set.")
    temp, det = _get_temperature(p, det)
    return OpenAIModel(name=name, api_key=key, model_id=p["model_id"], config=mc,
        deterministic=det, temperature=temp,
        max_retries=p.get("max_retries", 3),
        timeout=p.get("timeout", 60), max_tokens=p.get("max_tokens", 512),
        base_url=p.get("base_url", None))


def _build_anthropic(name, p, mc, det):
    from models.anthropic_model import AnthropicModel
    key = os.environ.get(p.get("api_key_env", ""), "")
    if not key:
        raise EnvironmentError(f"Env var '{p.get('api_key_env')}' not set.")
    temp, det = _get_temperature(p, det)
    return AnthropicModel(name=name, api_key=key, model_id=p["model_id"], config=mc,
        deterministic=det, temperature=temp,
        max_retries=p.get("max_retries", 3),
        timeout=p.get("timeout", 60), max_tokens=p.get("max_tokens", 512))


def _build_gemini(name, p, mc, det):
    from models.gemini_model import GeminiModel
    key = os.environ.get(p.get("api_key_env", ""), "")
    if not key:
        raise EnvironmentError(f"Env var '{p.get('api_key_env')}' not set.")
    temp, det = _get_temperature(p, det)
    return GeminiModel(name=name, api_key=key, model_id=p["model_id"], config=mc,
        deterministic=det, temperature=temp,
        max_retries=p.get("max_retries", 5),
        timeout=p.get("timeout", 60), max_tokens=p.get("max_tokens", 512),
        request_delay=p.get("request_delay", 0.5))


def _build_deepseek(name, p, mc, det):
    from models.deepseek_model import DeepSeekModel
    key = os.environ.get(p.get("api_key_env", ""), "")
    if not key:
        raise EnvironmentError(f"Env var '{p.get('api_key_env')}' not set.")
    temp, det = _get_temperature(p, det)
    return DeepSeekModel(name=name, api_key=key, model_id=p["model_id"], config=mc,
        base_url=p.get("base_url"), deterministic=det, temperature=temp,
        max_retries=p.get("max_retries", 3), timeout=p.get("timeout", 60),
        max_tokens=p.get("max_tokens", 512))


def _build_asu(name, p, mc, det):
    from models.asu_model import ASUCreateAIModel
    key = os.environ.get(p.get("api_key_env", "ASU_CREATEAI_TOKEN"), "")
    if not key:
        raise EnvironmentError(f"Env var '{p.get('api_key_env', 'ASU_CREATEAI_TOKEN')}' not set.")
    temp, det = _get_temperature(p, det)
    base_url = p.get("base_url") or os.environ.get("ASU_BASE_URL", None)
    return ASUCreateAIModel(name=name, api_key=key,
        model_name=p["model_name"], model_provider=p["model_provider"],
        config=mc, base_url=base_url, deterministic=det, temperature=temp,
        max_retries=p.get("max_retries", 5), timeout=p.get("timeout", 120),
        max_tokens=p.get("max_tokens", 512),
        request_delay=p.get("request_delay", 1.0),
        system_prompt=p.get("system_prompt", ""))


def _build_local(name, p, mc, det):
    from models.local_model import LocalModel
    # If temperature explicitly given in params, force stochastic sampling
    temperature = p.get("temperature", None)
    if temperature is not None and float(temperature) > 0:
        det = False
    return LocalModel(name=name, model_id=p["model_id"], config=mc,
        device=p.get("device", "cpu"),
        max_new_tokens=p.get("max_new_tokens", 256),
        use_4bit=p.get("use_4bit", None),
        trust_remote_code=p.get("trust_remote_code", True),
        temperature=temperature,
        deterministic=det)


def _build_mock(name, p, mc, det, seed):
    from models.mock_model import MockModel
    return MockModel(name=name, config=mc,
        accuracy_level=p.get("accuracy_level", 0.8),
        seed=seed, deterministic=det)


# -- Model registry ---------------------------------------------
# To add a new model type:
#   1. Create models/my_model.py  (extend BaseModel)
#   2. Add one line here:  "my_type": lambda n,p,mc,det,s: _build_my(n,p,mc,det)
# ---------------------------------------------------------------
def _get_model_registry(seed):
    return {
        "openai":    lambda n, p, mc, det, s: _build_openai(n, p, mc, det),
        "anthropic": lambda n, p, mc, det, s: _build_anthropic(n, p, mc, det),
        "gemini":    lambda n, p, mc, det, s: _build_gemini(n, p, mc, det),
        "deepseek":  lambda n, p, mc, det, s: _build_deepseek(n, p, mc, det),
        "asu":       lambda n, p, mc, det, s: _build_asu(n, p, mc, det),
        "local":     lambda n, p, mc, det, s: _build_local(n, p, mc, det),
        "mock":      lambda n, p, mc, det, s: _build_mock(n, p, mc, det, s),
    }


# -------------------------------------------------------------
# Registry: dataset type -> builder function
# Adding a new dataset: add one entry here, create the class
# -------------------------------------------------------------

def _build_synthetic(name, p):
    from llm_datasets.synthetic_dataset import SyntheticDataset
    return SyntheticDataset(name=name, config=p,
        num_reasoning=p.get("num_reasoning", 20),
        num_adversarial=p.get("num_adversarial", 10),
        num_robustness=p.get("num_robustness", 10),
        seed=p.get("seed", 42))


def _build_gsm8k(name, p):
    from llm_datasets.gsm8k_dataset import GSM8KDataset
    return GSM8KDataset(name=name, config=p,
        num_samples=p.get("num_samples", 250),
        seed=p.get("seed", 42))


def _build_strategyqa(name, p):
    from llm_datasets.strategyqa_dataset import StrategyQADataset
    return StrategyQADataset(name=name, config=p,
        num_samples=p.get("num_samples", 250),
        seed=p.get("seed", 42))


def _build_mmlu(name, p):
    from llm_datasets.mmlu_dataset import MMLUDataset
    return MMLUDataset(name=name, config=p,
        num_samples=p.get("num_samples", 250),
        seed=p.get("seed", 42),
        subjects=p.get("subjects", None))


def _build_json_dataset(name, p):
    from llm_datasets.base_dataset import BaseDataset
    return BaseDataset.from_json(
        path=p["path"], name=name, seed=p.get("seed", 42))


# -- Dataset registry -------------------------------------------
# To add a new dataset:
#   1. Create datasets/my_dataset.py  (extend BaseDataset)
#   2. Add one line here:  "my_type": lambda n, p: _build_my(n, p)
# ---------------------------------------------------------------
DATASET_REGISTRY = {
    "synthetic":  lambda n, p: _build_synthetic(n, p),
    "gsm8k":      lambda n, p: _build_gsm8k(n, p),
    "strategyqa": lambda n, p: _build_strategyqa(n, p),
    "mmlu":       lambda n, p: _build_mmlu(n, p),
    "json":       lambda n, p: _build_json_dataset(n, p),
}


# -------------------------------------------------------------
# Builders
# -------------------------------------------------------------

def build_models(config: dict) -> list:
    exp      = config.get("experiment", {})
    seed     = exp.get("seed", 42)
    det      = exp.get("deterministic", True)
    registry = _get_model_registry(seed)
    models   = []

    for mc in config.get("models", []):
        name  = mc["name"]
        mtype = mc.get("type", "mock")
        p     = mc.get("params", {})

        if mtype not in registry:
            logger.warning(f"  X Unknown model type '{mtype}' for '{name}'. "
                           f"Available: {list(registry.keys())}")
            continue
        try:
            model = registry[mtype](name, p, mc, det, seed)
            models.append(model)
            logger.info(f"  OK {name:32s} [{mtype}]")
        except EnvironmentError as e:
            logger.warning(f"  ! {name:32s} skipped -- {e}")
        except Exception as e:
            logger.error(f"  X {name:32s} failed  -- {e}")

    return models


def build_dataset(config: dict, output_dir: str):
    from llm_datasets.multi_dataset import MultiDataset
    seed     = config.get("experiment", {}).get("seed", 42)
    combined = MultiDataset(name="combined", seed=seed)

    for ds_cfg in config.get("datasets", []):
        ds_type = ds_cfg.get("type", "synthetic")
        name    = ds_cfg.get("name", ds_type)
        p       = ds_cfg.get("params", {})

        if ds_type not in DATASET_REGISTRY:
            logger.warning(f"  X Unknown dataset type '{ds_type}'. "
                           f"Available: {list(DATASET_REGISTRY.keys())}")
            continue
        try:
            ds = DATASET_REGISTRY[ds_type](name, p)
            ds.load()
            combined.add(ds)
            logger.info(f"  OK {name:20s} [{ds_type}] -- {len(ds):>5} items")
            ds.save_json(os.path.join(output_dir, f"dataset_{name}.json"))
        except Exception as e:
            logger.error(f"  X Dataset '{name}' failed -- {e}")

    combined.load()
    return combined


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def print_banner():
    print("""
+==================================================================+
|   LLM Reasoning Quality Evaluation Framework  (v5)              |
|   Config-driven: models . datasets . weights -- no code needed   |
|   Metrics: CQ . CS . RS . LS . ES . SS                          |
+==================================================================+""")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=None,
        help="Path to config YAML (default: config/config.yaml)"
    )
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = args.config or os.path.join(base_dir, "config", "config.yaml")

    if not os.path.exists(cfg_path):
        logger.error(f"Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp = config.get("experiment", {})
    set_seed(exp.get("seed", 42))
    if exp.get("deterministic", True):
        enable_deterministic_mode()

    experiment_id = f"{exp.get('name','exp')}_{int(time.time())}"
    output_dir    = os.path.join(base_dir, exp.get("output_dir", "outputs"))
    os.makedirs(output_dir, exist_ok=True)

    _set_log_level(config)
    logger.info(f"Config        : {cfg_path}")
    logger.info(f"Experiment ID : {experiment_id}")
    logger.info(f"Output dir    : {output_dir}\n")

    # -- Models --------------------------------------------------
    logger.info("Registering models...")
    models = build_models(config)
    if not models:
        logger.error(
            "\nNo models loaded. Check:\n"
            "  * API key env vars are set (or .env file exists)\n"
            "  * HuggingFace model IDs are correct\n"
            "  * pip install -r requirements.txt\n"
        )
        sys.exit(1)
    logger.info(f"\n{len(models)} model(s) ready.\n")

    # -- Datasets ------------------------------------------------
    logger.info("Loading datasets...")
    dataset = build_dataset(config, output_dir)
    logger.info(f"\nDataset summary : {dataset.summary()}")
    logger.info(f"Total items     : {len(dataset)}\n")

    # -- Evaluate ------------------------------------------------
    from evaluation.evaluator import Evaluator
    evaluator = Evaluator(
        config=config, output_dir=output_dir, experiment_id=experiment_id)
    results = evaluator.evaluate_all(models=models, dataset=dataset)

    # -- Outputs -------------------------------------------------
    excel_path = evaluator.tracker.export_excel(
        results, "reasoning_quality_results.xlsx")

    from visualization.radar_plot import RadarPlot
    exp_dir    = os.path.join(output_dir, experiment_id)
    plotter    = RadarPlot(output_dir=exp_dir)
    radar_path = plotter.plot(results, "radar_plot.png")
    bar_path   = plotter.plot_bar_comparison(results, "bar_comparison.png")

    # -- Summary table -------------------------------------------
    print("\n" + "="*84)
    print(f" {'Model':<24} {'CQ':>6} {'CS':>6} {'RS':>6} {'LS':>6} {'ES':>6} {'SS':>6}  {'Balanced':>8}")
    print("-"*84)
    for r in results:
        m   = r["raw_metrics"]
        agg = r.get("aggregated", {})
        print(
            f" {r['model']:<24} "
            f"{m.get('correctness',0):.3f}  "
            f"{m.get('consistency',0):.3f}  "
            f"{m.get('robustness',0):.3f}  "
            f"{m.get('logical_coherence',0):.3f}  "
            f"{m.get('efficiency',0):.3f}  "
            f"{m.get('stability',0):.3f}   "
            f"{agg.get('balanced',0):.3f}"
        )
    print("="*84)
    print("  CQ=Correctness . CS=Consistency . RS=Robustness")
    print("  LS=Logical Coherence . ES=Efficiency . SS=Stability\n")
    print(f"  Excel   -> {excel_path}")
    print(f"  Radar   -> {radar_path}")
    print(f"  Results -> {exp_dir}\n")

    return results


if __name__ == "__main__":
    main()
