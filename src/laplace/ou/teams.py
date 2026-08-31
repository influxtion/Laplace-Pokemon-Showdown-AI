r"""Our own OU team: find it, check it, hand it to poke-env.

In Random Battle the team is generated and the bot has nothing to bring. In OU the team is
half the bot -- the same search plays very differently behind a hyper-offence lead and
behind a Great Tusk / Clefable balance -- so teams are committed to `teams/` at the repo
root, one Showdown export per `.txt` file, and referred to by their file stem:

    python -m laplace.cli.ladder --format gen9ou --team balance

poke-env accepts a Showdown export directly and packs it itself (ConstantTeambuilder), so
this module deliberately does NOT re-implement the packer. What it adds is the check:
Showdown rejects a malformed team at MATCH time, which means a bad file shows up as a
mysterious failure several minutes into a run rather than as an error at startup. `load`
parses and counts before a single game is queued.
"""

import os

from poke_env.teambuilder import Teambuilder

from laplace import paths

TEAM_SIZE = 6
_SUFFIX = ".txt"


class TeamError(Exception):
    """A team file that cannot be used. Raised at startup, never mid-run."""


def normalize(text):
    r"""Clean a Showdown export so poke-env's parser can read it.

    Not cosmetic -- poke-env splits an export into Pokemon on the literal string "\n\n" and
    then parses each block line by line, so anything other than bare LF endings breaks it,
    and it breaks in the two worst ways:

      "\r\n"    a team file saved by any Windows editor. Crashes with KeyError('spe\r')
                while parsing the EV line -- and since load() hands this text straight to
                poke-env's ConstantTeambuilder, the crash lands mid-battle rather than at
                startup. This is the default line ending on the machine this runs on.
      "\r\r\n"  what pokepast.es/<id>/raw actually serves, which is the normal way anyone
                gets a sample team. Same crash.
      trailing  pokepaste also pads every line with two spaces, so the blank separator line
      spaces    is "  " and the "\n\n" split silently finds ONE Pokemon instead of six.
                Silent is worse than the crash: it surfaces as a confusing count error.

    Fixing it here rather than asking people to convert their files is the right layer: the
    export is what everyone has, and it is a text format we consume, not one we define."""
    # CR is DELETED, not translated. Translating it turns pokepaste's "\r\r\n" into two
    # newlines, which is the block separator -- so every single line becomes its own
    # Pokemon and a six-mon team parses as ten. The only case where a CR carries meaning
    # is a classic-Mac file with no LF at all, handled first.
    if "\n" not in text and "\r" in text:
        text = text.replace("\r", "\n")
    text = text.replace("\r", "")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip("\n")
    while "\n\n\n" in text:          # blank separator lines can now be doubled up
        text = text.replace("\n\n\n", "\n\n")
    return text + "\n"


def team_dir():
    return paths.TEAMS_DIR


def available():
    """Team names (file stems) in teams/, sorted."""
    try:
        return sorted(f[:-len(_SUFFIX)] for f in os.listdir(paths.TEAMS_DIR)
                      if f.endswith(_SUFFIX))
    except OSError:
        return []


def resolve(name):
    """A --team argument -> a path. Accepts a bare name, a stem, or a real path."""
    if not name:
        raise TeamError("no team given")
    for candidate in (name, name + _SUFFIX,
                      os.path.join(paths.TEAMS_DIR, name),
                      os.path.join(paths.TEAMS_DIR, name + _SUFFIX)):
        if os.path.isfile(candidate):
            return candidate
    known = available()
    listing = ", ".join(known) if known else f"none yet -- put one in {paths.TEAMS_DIR}"
    raise TeamError(f"no team file for {name!r} (looked in {paths.TEAMS_DIR}; have: {listing})")


def load(name):
    """(showdown-export text, parsed members) for a team, or raise TeamError.

    The text is returned unpacked because that is what poke-env wants, but NORMALIZED --
    poke-env's own parser cannot read a CRLF export (see normalize), and this text goes
    straight into its teambuilder, so handing back the raw bytes would just move the crash
    to the first battle. The parse is done only to fail loudly here instead of quietly on
    the ladder."""
    path = resolve(name)
    try:
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
    except OSError as exc:
        raise TeamError(f"could not read {path}: {exc}") from exc
    text = normalize(text)
    return text, validate(text, source=path)


def validate(text, source="<team>"):
    """Parse a Showdown export and check it is a well-formed singles team.

    WELL-FORMED, not tier-legal: this cannot tell you Tera Blast is banned. Showdown's own
    validator decides that when the match is made -- see teams/README.md for how to check a
    team offline, and PopupWatcher in laplace/cli/ladder.py for what a rejection looks like
    at run time."""
    text = normalize(text)
    try:
        members = Teambuilder.parse_showdown_team(text)
    except Exception as exc:
        raise TeamError(f"{source}: not a Showdown export ({exc!r}). Copy it out of the "
                        f"teambuilder with 'Import/Export'.") from exc
    if len(members) != TEAM_SIZE:
        raise TeamError(f"{source}: {len(members)} Pokemon, expected {TEAM_SIZE}")
    try:
        packed = Teambuilder.join_team(members)
    except Exception as exc:
        raise TeamError(f"{source}: could not be packed ({exc!r})") from exc
    if not packed:
        raise TeamError(f"{source}: packed to nothing")
    return members


def describe(members):
    """'Great Tusk / Gholdengo / ...' for the startup banner.

    species-then-nickname, not the other way round: poke-env's parser leaves `species` empty
    and puts the species in `nickname` for an un-nicknamed Pokemon, so reading the nickname
    first prints real nicknames for the ones that have them and nothing useful for the rest.
    """
    return " / ".join(m.species or m.nickname or "?" for m in members)
