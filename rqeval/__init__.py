"""
LLM Reasoning Quality Evaluation Framework
==========================================
pip install rqeval

Quick start:
    rqeval setup
    rqeval --config config/config_test.yaml

Paper: https://arxiv.org/abs/2605.24661
"""

__version__ = "1.2.0"
__author__  = "Ali Şenol, Garima Agrawal, Huan Liu"


def run_evaluation(config_path: str = "config/config.yaml",
                   output_dir: str = None) -> list:
    import os, sys, importlib.util

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    main_path = os.path.join(cwd, "main.py")
    if not os.path.exists(main_path):
        raise RuntimeError(
            f"main.py not found in {cwd}.\n"
            "Run 'rqeval setup' first to set up your project directory."
        )

    spec = importlib.util.spec_from_file_location("_llm_main", main_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sys.argv = ["rqeval", "--config", config_path]
    if output_dir:
        sys.argv += ["--output", output_dir]

    return mod.main()
