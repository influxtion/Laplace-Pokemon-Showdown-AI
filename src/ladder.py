r"""Play the trained bot on the real Pokemon Showdown ladder against humans.

Unlike every other script here, this connects to the OFFICIAL server (sim3.psim.us), not
the local one -- so the local server does NOT need to be running. By default it plays the
agent we ship (the policy + 1-ply search) as a registered account; pass --raw for the bare
policy.

Setup (once):
  1. Go to https://play.pokemonshowdown.com, click the gear (Options) -> Register, and
     register the username you want the bot to play as, with a password.
  2. Credentials come from a .env file in the project root (gitignored), or from the
     SHOWDOWN_USERNAME / SHOWDOWN_PASSWORD environment variables, or --username/--password.
     The .env format is two lines:
         SHOWDOWN_USERNAME=Influxobot
         SHOWDOWN_PASSWORD=your-password

Then, from the project root:
    python -u src\ladder.py                       # play 10 ranked ladder games
    python -u src\ladder.py --battles 25          # play 25
    python -u src\ladder.py --raw                 # bare policy, no search
    python -u src\ladder.py --mode accept         # wait for someone to challenge you
    python -u src\ladder.py --mode challenge --opponent SomeUser

Ladder games are rated, so your GXE/ELO shows on the account's profile -- the honest test
the training scripts couldn't give us. Follow Showdown's rules for automated play: keep the
account present, don't flood the ladder, and don't farm a specific person.
"""

import argparse
import asyncio
import os

from sb3_contrib import MaskablePPO

from poke_env import AccountConfiguration, ShowdownServerConfiguration

from eval_search import pick_model_path
from search import SearchPlayer
from opponent import ModelPlayer

BATTLE_FORMAT = "gen9randombattle"

# .env lives in the project root, one level up from this src/ file.
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def load_dotenv(path=_ENV_PATH):
    """Read KEY=value lines from .env into the environment (real env vars win)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def build_agent(model, account, use_search):
    """The bot to ladder with: the search wrapper (default) or the bare policy (--raw),
    pointed at the official server under the given registered account."""
    kwargs = dict(
        account_configuration=account,
        server_configuration=ShowdownServerConfiguration,
        battle_format=BATTLE_FORMAT,
    )
    if use_search:
        return SearchPlayer(model=model, **kwargs)
    return ModelPlayer(model=model, deterministic=True, **kwargs)


async def main():
    load_dotenv()  # populate SHOWDOWN_* from .env before reading them as argparse defaults
    parser = argparse.ArgumentParser(description="Play the bot on the real Showdown ladder.")
    parser.add_argument("--mode", choices=("ladder", "accept", "challenge"), default="ladder",
                        help="ladder = ranked games; accept = wait for a challenge; "
                             "challenge = challenge --opponent")
    parser.add_argument("--battles", type=int, default=10, help="how many games to play")
    parser.add_argument("--opponent", default=None, help="username to challenge (challenge mode)")
    parser.add_argument("--raw", action="store_true",
                        help="use the bare policy instead of the 1-ply search wrapper")
    parser.add_argument("--model", default=None,
                        help="model .zip to load (default: best available, see eval_search)")
    parser.add_argument("--username", default=os.environ.get("SHOWDOWN_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("SHOWDOWN_PASSWORD"))
    args = parser.parse_args()

    if not args.username:
        parser.error("no username: set SHOWDOWN_USERNAME or pass --username "
                     "(register it first at play.pokemonshowdown.com).")
    if args.mode == "challenge" and not args.opponent:
        parser.error("challenge mode needs --opponent USERNAME.")

    account = AccountConfiguration(args.username, args.password)
    model_path = args.model or pick_model_path()
    label = "raw policy" if args.raw else "search agent"
    print(f"Loading {model_path}; playing as '{args.username}' ({label}).", flush=True)
    model = MaskablePPO.load(model_path)

    agent = build_agent(model, account, use_search=not args.raw)

    if args.mode == "ladder":
        print(f"Queuing for {args.battles} ranked ladder game(s)...", flush=True)
        await agent.ladder(args.battles)
    elif args.mode == "accept":
        who = args.opponent or "anyone"
        print(f"Waiting for {args.battles} challenge(s) from {who} "
              f"(challenge '{args.username}' in {BATTLE_FORMAT})...", flush=True)
        await agent.accept_challenges(args.opponent, args.battles)
    else:
        print(f"Challenging {args.opponent} to {args.battles} game(s)...", flush=True)
        await agent.send_challenges(args.opponent, args.battles)

    won = agent.n_won_battles
    total = len(agent.battles)
    print(f"\nDone: won {won}/{total} ({won / total:.0%})." if total else "\nNo games played.",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
