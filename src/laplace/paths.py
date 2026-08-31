"""Every on-disk location the project reads or writes, resolved once.

Anchored to the project root via ``__file__``, never the working directory, so
``python -m laplace.cli.ladder`` from anywhere writes to the one replay archive
and reads the one copy of the Random Battle data.

This module is the *only* place that knows how deep the package sits under the
repo root. Everything else imports these names, so moving the package is a
one-line change here instead of a hunt through a dozen modules.
"""

import os

# src/laplace/paths.py -> src/laplace -> src -> <repo root>
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- committed inputs ---------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "data")
JOINT_SETS = os.path.join(DATA_DIR, "joint_sets_gen9.json")
RANDBATS_STATS = os.path.join(DATA_DIR, "randbats_stats_gen9.json")

MODELS_DIR = os.path.join(ROOT, "models")
VALUE_NET = os.path.join(MODELS_DIR, "value_net.pt")

# --- cloned/built during setup, not committed ---------------------------------
SERVER_DIR = os.path.join(ROOT, "server")
RANDBATS_SETS = os.path.join(SERVER_DIR, "data", "random-battles", "gen9", "sets.json")

FORK_DIR = os.path.join(ROOT, "poke-engine-fork")
FORK_WHEEL = os.path.join(FORK_DIR, "py")
FORK_VALUE_NET = os.path.join(FORK_DIR, "value_net_v5.bin")

# --- generated ----------------------------------------------------------------
REPLAYS_DIR = os.path.join(ROOT, "replays")
LADDER_REPLAY_DIR = os.path.join(REPLAYS_DIR, "ladder")
LADDER_OLD_REPLAY_DIR = os.path.join(REPLAYS_DIR, "ladder_old")
FOULPLAY_REPLAY_DIR = os.path.join(REPLAYS_DIR, "foulplay")

VALUE_DATA_DIR = os.path.join(ROOT, "data_value2")   # v2 features (368)

# --- local secrets ------------------------------------------------------------
ENV_FILE = os.path.join(ROOT, ".env")

# --- Gen 9 OU -----------------------------------------------------------------
# Built by `python -m laplace.cli.fetch_ou_data`, committed like the randbats data.
OU_DATA_DIR = os.path.join(DATA_DIR, "ou")
OU_USAGE = os.path.join(OU_DATA_DIR, "usage_gen9ou.json")
OU_SETS = os.path.join(OU_DATA_DIR, "sets_gen9ou.json")

# Our own OU teams: Showdown-export .txt files, one team per file. Committed --
# unlike Random Battle, the team IS part of the bot.
TEAMS_DIR = os.path.join(ROOT, "teams")

OU_REPLAY_DIR = os.path.join(REPLAYS_DIR, "ladder_ou")
