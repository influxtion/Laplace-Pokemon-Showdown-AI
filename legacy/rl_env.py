"""The learning environment. Stage two of the project.

Stage one was writing rules by hand. Here we do the opposite: describe the battle to a
network as a long list of numbers, tell it how well it's doing, and let it work out a
strategy by playing thousands of games.

The library handles the server, the legal moves, and turning the network's choice into an
actual command. We supply the two pieces that are specific to us:

  * the observation: the battle, as numbers
  * the reward: how good our position is

The observation is deliberately detailed. An early version used just 12 numbers and got
stuck, because it was blind to most of the game. This one includes the things that
actually drive a decision: boosts, status, the bench, move details, terrain, screens,
hazards, weather, what the opponent has shown, and the abilities that matter.
"""

import numpy as np
from gymnasium.spaces import Box

from poke_env.environment.singles_env import SinglesEnv

from knowledge import KNOWLEDGE, ROLE_NAMES, N_THREAT_FLAGS, estimate_damage_fraction, safe_priority

# --- the vocabulary ----------------------------------------------------------

STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe"]
STATUS_NAMES = ["BRN", "PAR", "SLP", "FRZ", "PSN", "TOX"]
EFFECT_NAMES = ["SUBSTITUTE", "LEECH_SEED", "TAUNT", "CONFUSION"]
TYPE_NAMES = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING", "POISON",
    "GROUND", "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST", "DRAGON", "DARK", "STEEL", "FAIRY",
]

# Items that change how a turn plays out, one flag per group. Only used for our own
# Pokemon, since we usually can't see what the opponent is holding and a column of zeroes
# teaches the network nothing.
ITEM_CATEGORIES = [
    {"leftovers", "blacksludge"},                  # heals a bit every turn
    {"choiceband", "choicespecs", "choicescarf"},  # more power or speed, locked to one move
    {"heavydutyboots"},                            # walks over hazards
    {"lifeorb"},                                    # hits harder, hurts itself
    {"assaultvest"},                               # tanky, but can't use status moves
    {"focussash"},                                 # survives one hit from full health
    {"rockyhelmet"},                               # hurts anything that touches it
    {"eviolite"},                                  # bulk for the unevolved
    {"boosterenergy"},                             # switches on a stat boost immediately
    {"weaknesspolicy"},                            # gets stronger after a super-effective hit
]

# Same idea for abilities. Grouping them by what they do, rather than listing hundreds of
# names, keeps this small while still capturing the ones that change a decision.
ABILITY_CATEGORIES = [
    {"levitate"},                                      # can't be hit by Ground
    {"flashfire", "wellbakedbody"},                    # can't be hit by Fire
    {"waterabsorb", "stormdrain", "dryskin"},          # can't be hit by Water
    {"voltabsorb", "lightningrod", "motordrive"},      # can't be hit by Electric
    {"sapsipper"},                                     # can't be hit by Grass
    {"multiscale", "shadowshield"},                    # takes half damage at full health
    {"intimidate"},                                    # weakens whatever it faces
    {"regenerator"},                                   # heals every time it switches out
    {"magicguard"},                                    # ignores poison, hazards, weather
    {"unaware"},                                       # ignores the opponent's boosts
    {"hugepower", "purepower"},                        # double attack
    {"speedboost", "protosynthesis", "quarkdrive",     # gets faster, one way or another
     "swiftswim", "chlorophyll", "sandrush", "unburden"},
]


# --- turning one Pokemon into numbers ----------------------------------------

def _base_stats(mon):
    """The six base stats, scaled down to roughly 0-1."""
    if mon is None:
        return [0.0] * 6
    return [mon.base_stats[k] / 200.0 for k in STAT_KEYS]


def _boosts(mon):
    """Stat boosts, squashed into -1 to 1, so a maxed-out Swords Dance reads as 1."""
    if mon is None:
        return [0.0] * 5
    return [mon.boosts[k] / 6.0 for k in BOOST_KEYS]


def _status_onehot(mon):
    """Which status condition it has, if any. All zeroes means healthy."""
    vec = [0.0] * len(STATUS_NAMES)
    if mon is not None and mon.status is not None and mon.status.name in STATUS_NAMES:
        vec[STATUS_NAMES.index(mon.status.name)] = 1.0
    return vec


def _effects(mon):
    """A few lingering effects worth knowing about: Substitute, Leech Seed and friends."""
    vec = [0.0] * len(EFFECT_NAMES)
    if mon is not None:
        names = {e.name for e in mon.effects}
        for i, nm in enumerate(EFFECT_NAMES):
            if nm in names:
                vec[i] = 1.0
    return vec


def _norm_ability(name):
    return name.lower().replace(" ", "").replace("-", "")


def _ability_flags(mon, infer_possible=False):
    """One number per ability group: 1 if we know it has one, 0.5 if it might, 0 if not.

    The half-marks are for the opponent, whose ability we usually haven't seen but whose
    species tells us what it could be.
    """
    vec = [0.0] * len(ABILITY_CATEGORIES)
    if mon is None:
        return vec
    if mon.ability:  # known for certain, which is always the case for our own team
        aid = _norm_ability(mon.ability)
        for i, ids in enumerate(ABILITY_CATEGORIES):
            if aid in ids:
                vec[i] = 1.0
    elif infer_possible:  # the opponent hasn't shown theirs yet
        possibles = {_norm_ability(a) for a in (mon.possible_abilities or [])}
        for i, ids in enumerate(ABILITY_CATEGORIES):
            if possibles & ids:
                vec[i] = 0.5
    return vec


def _matchup(me, opp):
    """Who has the better typing: our best matchup into them, and theirs into us."""
    if me is None or opp is None:
        return [0.0, 0.0]
    offense = max((opp.damage_multiplier(t) for t in me.types if t is not None), default=1.0)
    defense = max((me.damage_multiplier(t) for t in opp.types if t is not None), default=1.0)
    return [offense / 4.0, defense / 4.0]


def _eff_speed(mon):
    """How fast something actually is right now, counting boosts and paralysis.

    The original version just compared base speed, which is wrong the moment anything gets
    boosted or paralysed. A Choice Scarf is still invisible to us, since we can't see the
    opponent's item.
    """
    if mon is None:
        return 0.0
    boost = mon.boosts["spe"]
    mult = (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)
    spe = mon.base_stats["spe"] * mult
    if mon.status is not None and mon.status.name == "PAR":
        spe *= 0.5
    return spe


def _item_flags(mon):
    """Which of the notable item groups this Pokemon's item falls into."""
    vec = [0.0] * len(ITEM_CATEGORIES)
    if mon is None or not mon.item:
        return vec
    item = mon.item.lower().replace(" ", "").replace("-", "")
    for i, ids in enumerate(ITEM_CATEGORIES):
        if item in ids:
            vec[i] = 1.0
    return vec


def _tera_features(battle, me):
    """Our Tera situation: is it still available, what type, and have we used it.

    All of this is known for our own side. Theirs stays hidden until they use it, so
    there's nothing to record.
    """
    can_tera = 1.0 if battle.can_tera else 0.0
    if me is not None and me.tera_type is not None:
        tera_type = [1.0 if name == me.tera_type.name else 0.0 for name in TYPE_NAMES]
    else:
        tera_type = [0.0] * len(TYPE_NAMES)
    is_tera = 1.0 if (me is not None and me.is_terastallized) else 0.0
    return [can_tera] + tera_type + [is_tera]


# --- turning the rest of the field into numbers ------------------------------

def _hazards(side_conditions):
    """Entry hazards on one side of the field."""
    sr, spikes, tspikes = 0.0, 0.0, 0.0
    for cond, value in side_conditions.items():
        name = cond.name
        if name == "STEALTH_ROCK":
            sr = 1.0
        elif name == "SPIKES":
            spikes = value / 3.0
        elif name == "TOXIC_SPIKES":
            tspikes = value / 2.0
    return [sr, spikes, tspikes]


def _screens(side_conditions):
    """Screens and Tailwind on one side of the field."""
    refl = ls = av = tw = 0.0
    for cond in side_conditions:
        name = cond.name
        if name == "REFLECT":
            refl = 1.0
        elif name == "LIGHT_SCREEN":
            ls = 1.0
        elif name == "AURORA_VEIL":
            av = 1.0
        elif name == "TAILWIND":
            tw = 1.0
    return [refl, ls, av, tw]


def _terrain(battle):
    """Which terrain is up, if any."""
    elec = grass = psy = mist = 0.0
    for f in battle.fields:
        name = f.name
        if name == "ELECTRIC_TERRAIN":
            elec = 1.0
        elif name == "GRASSY_TERRAIN":
            grass = 1.0
        elif name == "PSYCHIC_TERRAIN":
            psy = 1.0
        elif name == "MISTY_TERRAIN":
            mist = 1.0
    none = 0.0 if (elec or grass or psy or mist) else 1.0
    return [elec, grass, psy, mist, none]


def _trick_room(battle):
    """Is Trick Room up? It reverses turn order, so slower becomes faster."""
    return [1.0 if any(f.name == "TRICK_ROOM" for f in battle.fields) else 0.0]


def _weather(battle):
    """What the weather is doing."""
    sun = rain = sand = snow = 0.0
    for w in battle.weather:
        name = w.name
        if name in ("SUNNYDAY", "DESOLATELAND"):
            sun = 1.0
        elif name in ("RAINDANCE", "PRIMORDIALSEA"):
            rain = 1.0
        elif name == "SANDSTORM":
            sand = 1.0
        elif name in ("SNOW", "HAIL"):
            snow = 1.0
    none = 0.0 if battle.weather else 1.0
    return [sun, rain, sand, snow, none]


def _bench(battle):
    """Our bench: how healthy each one is, and how it matches up against what's out."""
    opp = battle.opponent_active_pokemon
    feats = []
    benched = [m for m in battle.team.values() if m is not battle.active_pokemon][:5]
    for mon in benched:
        feats += [mon.current_hp_fraction] + _matchup(mon, opp)
    while len(feats) < 15:
        feats.append(0.0)
    return feats


def _opponent_moves(battle):
    """The moves we've seen them use: how strong they are, and how much they hurt us."""
    me = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    power = [0.0] * 4
    effectiveness = [0.0] * 4
    if opp is not None:
        for i, move in enumerate(list(opp.moves.values())[:4]):
            power[i] = move.base_power / 100.0
            if me is not None:
                effectiveness[i] = me.damage_multiplier(move) / 4.0
    return power + effectiveness


def _knowledge_features(battle, me, opp):
    """What we can guess about the opponent's set, and how hard people are about to hit.

    All of it hangs off the opponent's active Pokemon, which we can always see, and our own
    moves, which we always know, so nothing here is ever blank."""
    # How much damage each of our moves would do: the "can I kill it?" signal.
    my_move_dmg = [0.0] * 4
    for i, move in enumerate(battle.available_moves[:4]):
        my_move_dmg[i] = estimate_damage_fraction(me, move, opp)
    best_out = max(my_move_dmg) if my_move_dmg else 0.0
    # The worst they could do to us, weighted by how likely they are to have it. A cue
    # to switch out.
    worst_in = KNOWLEDGE.predicted_incoming(opp, me)

    return (
        my_move_dmg + [best_out, worst_in]
        + KNOWLEDGE.predicted_coverage(opp)     # do they have an attack of each type?
        + KNOWLEDGE.role_flags(opp)             # what kind of set are they likely running?
        + KNOWLEDGE.threat_flags(opp)           # priority, recovery, hazards, setup, status
    )


# How many numbers the whole observation adds up to. It started at 141, then gained move
# priority, our item, Tera, and finally the 40-number knowledge layer, reaching 215.
KNOWLEDGE_FEATURES = 6 + 18 + len(ROLE_NAMES) + N_THREAT_FLAGS
N_FEATURES = 215


def build_observation(battle):
    """Describe the battle to the network as one long list of numbers.

    Kept out here rather than inside the environment so the self-play opponent can build
    the same list when it runs the model itself."""
    me = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    # Our own moves.
    move_power = [0.0] * 4
    move_multiplier = [0.0] * 4
    move_accuracy = [0.0] * 4
    move_physical = [0.0] * 4
    move_status = [0.0] * 4
    move_priority = [0.0] * 4
    for i, move in enumerate(battle.available_moves[:4]):
        move_power[i] = move.base_power / 100.0
        if opp is not None:
            move_multiplier[i] = opp.damage_multiplier(move) / 4.0
        acc = move.accuracy
        move_accuracy[i] = 1.0 if acc is True else float(acc)
        category = move.category.name
        move_physical[i] = 1.0 if category == "PHYSICAL" else 0.0
        move_status[i] = 1.0 if category == "STATUS" else 0.0
        # Priority beats speed, so it matters a lot. Clamped, so one rare -7 move can't
        # blow out the range everything else lives in.
        move_priority[i] = max(-1.0, min(1.0, safe_priority(move) / 3.0))

    my_hp = me.current_hp_fraction if me else 0.0
    opp_hp = opp.current_hp_fraction if opp else 0.0
    my_remaining = sum(1 for m in battle.team.values() if not m.fainted) / 6.0
    opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
    opp_remaining = (6 - opp_fainted) / 6.0
    # Who moves first, counting boosts and paralysis, and flipped by Trick Room.
    faster = 1.0 if _eff_speed(me) > _eff_speed(opp) else 0.0
    if any(f.name == "TRICK_ROOM" for f in battle.fields):
        faster = 1.0 - faster

    features = (
        move_power + move_multiplier + move_accuracy + move_physical + move_status  # 20
        + move_priority                                        # 4
        + [my_hp, opp_hp, my_remaining, opp_remaining]          # 4
        + _base_stats(me) + _base_stats(opp)                    # 12
        + _boosts(me) + _boosts(opp)                            # 10
        + _status_onehot(me) + _status_onehot(opp)              # 12
        + _matchup(me, opp) + [faster]                          # 3
        + _effects(me) + _effects(opp)                          # 8
        + _ability_flags(me) + _ability_flags(opp, infer_possible=True)  # 24
        + _bench(battle)                                        # 15
        + _opponent_moves(battle)                               # 8
        + _hazards(battle.side_conditions)                      # 3
        + _hazards(battle.opponent_side_conditions)             # 3
        + _terrain(battle) + _trick_room(battle)                # 6
        + _screens(battle.side_conditions)                      # 4
        + _screens(battle.opponent_side_conditions)             # 4
        + _weather(battle)                                      # 5
        + _item_flags(me)                                       # 10 (what we're holding)
        + _tera_features(battle, me)                            # 20 (our Tera situation)
        + _knowledge_features(battle, me, opp)                  # 40 (guesses about them)
    )
    return np.array(features, dtype=np.float32)


# How much each thing is worth. Winning dwarfs everything else on purpose, so the agent
# plays to win rather than to trade evenly. The smaller HP and faint rewards exist to give
# it something to learn from early on, before it ever wins anything.
DEFAULT_REWARD = dict(fainted_value=1.0, hp_value=0.5, status_value=0.1, victory_value=100.0)


class ShowdownSinglesEnv(SinglesEnv):
    def __init__(self, reward_weights=None, reward_schedule=None, switch_penalty=0.0, **kwargs):
        super().__init__(**kwargs)
        # The lower bound is negative because stat drops are.
        obs_space = Box(low=-1.0, high=4.0, shape=(N_FEATURES,), dtype=np.float32)
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}
        self.reward_weights = dict(DEFAULT_REWARD, **(reward_weights or {}))

        # Discourage panic switching. The PokeLLMon paper found that switching two turns in
        # a row correlates with losing: the agent keeps running from matchups instead of
        # committing to one, burning turns and taking hazard damage on every entry. So the
        # second voluntary switch in a row costs something. A single switch is fine and
        # never penalised. Set to 0 to turn this off.
        self.switch_penalty = switch_penalty
        self._switch_state = {}  # per battle: what was out last turn, and did we switch

        # Optionally fade out the small rewards over time. The problem this solves: the
        # per-turn HP and faint rewards fire constantly while a win only pays out once a
        # game, so the agent reliably learns to trade evenly rather than to win. Annealing
        # starts with the small rewards turned up, to get a competent policy off the ground,
        # then shrinks them until winning is the only thing that really counts.
        self.reward_schedule = reward_schedule
        if reward_schedule is not None:
            self._anneal_start = dict(DEFAULT_REWARD, **reward_schedule["start"])
            self._anneal_end = dict(DEFAULT_REWARD, **reward_schedule["end"])
            self._anneal_horizon = max(1, int(reward_schedule["horizon"]))
            self._reward_steps = 0

    def embed_battle(self, battle):
        return build_observation(battle)

    def _current_weights(self):
        """This turn's reward weights: either fixed, or somewhere along the fade-out."""
        if self.reward_schedule is None:
            return self.reward_weights
        frac = min(1.0, self._reward_steps / self._anneal_horizon)
        self._reward_steps += 1
        return {k: self._anneal_start[k] + frac * (self._anneal_end[k] - self._anneal_start[k])
                for k in self._anneal_start}

    def _panic_switch_penalty(self, battle):
        """Charge for switching twice in a row.

        A switch counts as voluntary if the Pokemon we left is still alive; replacing
        something that fainted is forced and never counts. Any turn where we don't switch
        resets the streak, so normal defensive switching is never punished."""
        if not self.switch_penalty:
            return 0.0
        state = self._switch_state.setdefault(battle, {"prev_mon": None, "prev_voluntary": False})
        cur, prev = battle.active_pokemon, state["prev_mon"]
        # Each team slot is the same object turn to turn, so this really means "same Pokemon".
        switched = prev is not None and cur is not None and cur is not prev
        voluntary = switched and not prev.fainted
        panic = voluntary and state["prev_voluntary"]
        state["prev_mon"], state["prev_voluntary"] = cur, voluntary
        return self.switch_penalty if panic else 0.0

    def calc_reward(self, battle):
        """How much better or worse our position got this turn, minus any switching
        penalty. The weights may be fading out across training, see above."""
        reward = self.reward_computing_helper(battle, **self._current_weights())
        return reward - self._panic_switch_penalty(battle)
