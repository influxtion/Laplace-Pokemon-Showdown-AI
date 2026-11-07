# Contributing to Laplace

Laplace plays Gen 9 Random Battle. Changes to the OU bot belong on the `OU` branch;
start random-team work from `main`.

## Start here

Browse [good first issues](https://github.com/influxtion/Laplace-Pokemon-Showdown-AI/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
Use [Discussions](https://github.com/influxtion/Laplace-Pokemon-Showdown-AI/discussions)
for setup questions and ideas; issues are for reproducible bugs and scoped work.

1. Fork and clone the repository, then create a branch for your change.
2. Install Python 3.12, Git, and the stable Rust toolchain. Windows also needs the
   Visual Studio C++ Build Tools. Run `python scripts/setup.py` from the checkout.
3. Run `.venv/Scripts/python -m unittest discover -s tests -v` on Windows, or
   `.venv/bin/python -m unittest discover -s tests -v` on Linux/macOS.
4. Make a focused change and open a pull request against the matching branch.

The tests are offline: no Showdown account, credentials, or live ladder games.
The setup script installs CPU PyTorch and builds the engine with Terastallization.
CI exercises the same setup on Linux and Windows. macOS is not CI-tested.

## What to include in a pull request

Explain the problem, the resulting behavior, and how you checked it. For a bug,
include a small regression test or a sanitized replay reproducer when practical.
For a search or policy change, include old-vs-new local battle results (at least
60 games, configuration, wins/losses, and a Wilson interval) before claiming an
improvement in playing strength. Documentation changes do not need battle runs.

For local battles, install Node.js 22+ and run `python scripts/setup.py --local-server`.
Start Showdown in a second terminal with `node pokemon-showdown start --no-security`
from `server/`, then use `python -m laplace.cli.eval_engine` inside the virtual
environment. The default benchmark is a regression check; it does not establish
strength against high-rated humans.

## Repository conventions

- Keep runtime code under `src/laplace/` and disk locations in `laplace.paths`.
- Keep the ladder configuration in `laplace.cli.ladder.build_agent`.
- Do not commit `.env`, account passwords, tokens, personal settings, or large
  generated replay/training archives. Check your staged diff before committing.
- Keep model feature definitions compatible with the data used to train them.
- Preserve attribution for third-party code and data.

Contributions are under the repository's [MIT license](LICENSE). Be respectful,
give actionable feedback, and explain assumptions when discussing results.
