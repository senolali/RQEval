"""
CLI entry point for rqeval package.

Commands:
    rqeval setup                         set up project in current directory
    rqeval --config config/config.yaml   run evaluation
    rqeval --version                     show version
    rqeval --help                        show help
"""

import argparse
import os
import shutil
import sys


def _get_data_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "_data")


def _run_setup(target_dir: str = None) -> None:
    target = target_dir or os.getcwd()
    data   = _get_data_dir()

    if not os.path.isdir(data):
        print(f"[ERROR] Bundled data not found at: {data}")
        print("        Try reinstalling: pip install --force-reinstall rqeval")
        sys.exit(1)

    print(f"\n  Setting up LLM Reasoning Quality Framework in:\n  {target}\n")

    copied  = []
    skipped = []

    for item in os.listdir(data):
        src = os.path.join(data, item)
        dst = os.path.join(target, item)

        if os.path.isdir(src):
            if os.path.exists(dst):
                skipped.append(item + "/")
            else:
                shutil.copytree(src, dst)
                copied.append(item + "/")
        else:
            if os.path.exists(dst):
                skipped.append(item)
            else:
                shutil.copy2(src, dst)
                copied.append(item)

    os.makedirs(os.path.join(target, "outputs"), exist_ok=True)

    print("  Copied:")
    for f in copied:
        print(f"      {f}")

    if skipped:
        print("\n  Already exists (not overwritten):")
        for f in skipped:
            print(f"      {f}")

    print("""
  Setup complete!

  Next steps:
    1. Copy .env.example to .env and add your API keys
    2. Quick test (no API keys needed):
         python main.py --config config/config_test.yaml
    3. Full evaluation:
         python main.py --config config/config.yaml

  Paper: https://arxiv.org/abs/2605.24661
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rqeval",
        description=(
            "RQEval -- LLM Reasoning Quality Evaluation\n"
            "Metrics: CQ, CS, RS, LS, ES, SS\n\n"
            "Commands:\n"
            "  rqeval setup                       Set up project in current directory\n"
            "  rqeval --config config/config.yaml Run evaluation\n\n"
            "Paper: https://arxiv.org/abs/2605.24661"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["setup"],
        help="'setup' — copy framework files to current directory",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Target directory for setup (default: current directory)",
    )
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version",
    )

    args = parser.parse_args()

    if args.version:
        from rqeval import __version__
        print(f"RQEval {__version__}")
        return

    if args.command == "setup":
        _run_setup(target_dir=args.dir)
        return

    if args.config:
        config_path = args.config

        if not os.path.isfile(config_path):
            print(f"[ERROR] Config file not found: {config_path}")
            print("        Run 'rqeval setup' first.")
            sys.exit(1)

        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)

        main_py = os.path.join(cwd, "main.py")
        if not os.path.isfile(main_py):
            print("[ERROR] main.py not found in current directory.")
            print("        Run 'rqeval setup' first.")
            sys.exit(1)

        sys.argv = ["main.py", "--config", config_path]

        import importlib.util
        spec = importlib.util.spec_from_file_location("_llm_main", main_py)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # exec_module() loads the file but does NOT set mod.__name__ to
        # "__main__", so main.py's own `if __name__ == "__main__": main()`
        # guard never fires. Call it explicitly.
        if hasattr(mod, "main") and callable(mod.main):
            mod.main()
        else:
            print("[ERROR] main.py has no callable main() function.")
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
