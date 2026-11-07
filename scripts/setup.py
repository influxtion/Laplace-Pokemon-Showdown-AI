"""Install Laplace from a checkout: python scripts/setup.py."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=ROOT):
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", default=".venv", help="environment directory (default: .venv)")
    parser.add_argument("--local-server", action="store_true", help="also install Node dependencies for local battles")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12, the supported and CI-tested version.")
    required = ["git", "cargo"] + (["node", "npm"] if args.local_server else [])
    missing = [name for name in required if not shutil.which(name)]
    if missing:
        parser.error("Install these prerequisites and reopen your terminal: " + ", ".join(missing))
    env = (ROOT / args.venv).resolve()
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not env.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    elif not python.is_file():
        parser.error(f"{env} is not a virtual environment; choose another --venv directory.")
    # Never replace an existing environment or overwrite credentials.
    run(python, "-c", "import sys; assert sys.version_info[:2] == (3, 12), 'Environment must use Python 3.12'")
    run(python, "-m", "pip", "install", "--upgrade", "pip")
    run(python, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    run(python, "-m", "pip", "install", "--no-binary=poke-engine", "--force-reinstall",
        "poke-engine==0.0.47",
        "--config-settings=build-args=--features poke-engine/terastallization --no-default-features")
    run(python, "-m", "pip", "install", "-e", ".")
    server = ROOT / "server"
    if not server.exists():
        run("git", "clone", "--depth", "1", "https://github.com/smogon/pokemon-showdown.git", server)
    sets = server / "data/random-battles/gen9/sets.json"
    if not sets.is_file():
        raise SystemExit(f"Missing {sets}. Existing server directory was left untouched.")
    if args.local_server:
        run(shutil.which("npm"), "install", cwd=server)
    run(python, "-m", "unittest", "discover", "-s", "tests", "-v")
    print(f'\nReady. Copy .env.example to .env and fill in your Showdown account.\n'
          f'Run: "{python}" -m laplace.cli.analyze_battle')


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
