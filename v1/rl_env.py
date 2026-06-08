"""The reinforcement-learning environment (Stage 2).

This is where "training an AI" actually happens. We don't write rules for which
move to pick (that was Stage 1). Instead we describe the battle to a neural
network as a list of numbers, hand back a reward when things go well, and let a
learning algorithm (PPO) figure out a good policy through trial and error.

poke-env does the heavy lifting: `SinglesEnv` already knows how to talk to the
Showdown server, what actions are legal, and how to translate a chosen action
number into an in-game move. We only have to supply the two things that are
specific to *our* AI:

  * embed_battle()  -> turn the battle state into numbers (the "observation")
  * calc_reward()   -> say how good the current state is (the "reward")

The observation is deliberately detailed. An early 12-number version plateaued
against a strong opponent because it was blind to most of the game. This version
adds the things that actually drive decisions: stat boosts, status, the bench
(for switching), move details, field/terrain/screens, hazards, weather, the
opponent's revealed moves, and high-impact abilities.
"""

import numpy as np
from gymnasium.spaces import Box

from poke_env.environment.singles_env import SinglesEnv

from knowledge import KNOWLEDGE, ROLE_NAMES, N_THREAT_FLAGS, estimate_damage_fraction, safe_priority

# --- lookup tables -----------------------------------------------------------

STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]   # base-stat dict keys
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe"]        # in-battle stat-stage keys
STATUS_NAMES = ["BRN", "PAR", "SLP", "FRZ", "PSN", "TOX"]  # status conditions we encode
EFFECT_NAMES = ["SUBSTITUTE", "LEECH_SEED", "TAUNT", "CONFUSION"]  # volatile effects
TYPE_NAMES = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING", "POISON",
    "GROUND", "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST", "DRAGON", "DARK", "STEEL", "FAIRY",
]  # 18, used for the (own-side) Tera type one-hot

# High-impact held items, grouped into categories (one flag each). We only read this for
# OUR OWN active Pokemon, whose item is always known -- the opponent's is usually hidden,
# so encoding it would just add blank features (the mistake the 854-obs experiment made).
ITEM_CATEGORIES = [
    {"leftovers", "blacksludge"},                  # passive healing
    {"choiceband", "choicespecs", "choicescarf"},  # locked into one move (power/speed)
    {"heavydutyboots"},                            # ignores entry hazards
    {"lifeorb"},                                    # power boost + recoil
    {"assaultvest"},                               # special bulk, no status moves
    {"focussash"},                                 # survive a KO from full HP
    {"rockyhelmet"},                               # punishes contact
    {"eviolite"},                                  # bulk for not-fully-evolved
    {"boosterenergy"},                             # triggers Proto/Quark
    {"weaknesspolicy"},                            # boosts after a super-effective hit
]  # 10

# High-impact abilities, grouped into strategic categories. Each category becomes one
# flag per active Pokemon. Encoding categories (not the hundreds of raw abilities) keeps
# this compact while capturing the abilities that actually change a decision.
ABILITY_CATEGORIES = [
    {"levitate"},                                      # immune to Ground
    {"flashfire", "wellbakedbody"},                    # immune to Fire
    {"waterabsorb", "stormdrain", "dryskin"},          # immune to Water
    {"voltabsorb", "lightningrod", "motordrive"},      # immune to Electric
    {"sapsipper"},                                     # immune to Grass
    {"multiscale", "shadowshield"},                    # halves damage at full HP
    {"intimidate"},                                    # drops attack on switch-in
    {"regenerator"},                                   # heals on switch
    {"magicguard"},                                    # no indirect damage
    {"unaware"},                                       # ignores stat boosts
    {"hugepower", "purepower"},                        # doubles attack
    {"speedboost", "protosynthesis", "quarkdrive",     # speed enablers
     "swiftswim", "chlorophyll", "sandrush", "unburden"},
]


# --- per-Pokemon feature helpers ---------------------------------------------

def _base_stats(mon):
    """6 base stats scaled to roughly 0-1 (0 if the Pokemon is unknown)."""
    if mon is None:
        return [0.0] * 6
    return [mon.base_stats[k] / 200.0 for k in STAT_KEYS]


def _boosts(mon):
    """5 stat-stage boosts in -1..1 (a +6 Swords Dance reads as +1 attack)."""
    if mon is None:
        return [0.0] * 5
    return [mon.boosts[k] / 6.0 for k in BOOST_KEYS]


def _status_onehot(mon):
    """One-hot of the active status condition (all zeros = healthy)."""
    vec = [0.0] * len(STATUS_NAMES)
    if mon is not None and mon.status is not None and mon.status.name in STATUS_NAMES:
        vec[STATUS_NAMES.index(mon.status.name)] = 1.0
    return vec


def _effects(mon):
    """Flags for a few important volatile effects (substitute, leech seed, etc.)."""
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
    """One flag per high-impact ability category.

    1.0 = the Pokemon is confirmed to have an ability in this category.
    0.5 = the ability is unknown (opponent, not yet revealed) but the species COULD
          have one in this category (from `possible_abilities`).
    0.0 = none / unknown with no possibility.
    """
    vec = [0.0] * len(ABILITY_CATEGORIES)
    if mon is None:
        return vec
    if mon.ability:  # confirmed (always true for our own Pokemon)
        aid = _norm_ability(mon.ability)
        for i, ids in enumerate(ABILITY_CATEGORIES):
            if aid in ids:
                vec[i] = 1.0
    elif infer_possible:  # opponent whose ability hasn't shown yet
        possibles = {_norm_ability(a) for a in (mon.possible_abilities or [])}
        for i, ids in enumerate(ABILITY_CATEGORIES):
            if possibles & ids:
                vec[i] = 0.5
    return vec


def _matchup(me, opp):
    """Best type effectiveness I have into them, and they have into me (0-1 each)."""
    if me is None or opp is None:
        return [0.0, 0.0]
    offense = max((opp.damage_multiplier(t) for t in me.types if t is not None), default=1.0)
    defense = max((me.damage_multiplier(t) for t in opp.types if t is not None), default=1.0)
    return [offense / 4.0, defense / 4.0]


def _eff_speed(mon):
    """Effective speed: base speed adjusted for the stat-stage boost and paralysis.

    The old `faster` flag compared *raw* base speed, which is wrong whenever there's a
    speed boost/drop or paralysis in play. This captures those (the common cases);
    items like Choice Scarf are still invisible (held items aren't known for the opponent).
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
    """One flag per high-impact item category for a Pokemon whose item is known."""
    vec = [0.0] * len(ITEM_CATEGORIES)
    if mon is None or not mon.item:
        return vec
    item = mon.item.lower().replace(" ", "").replace("-", "")
    for i, ids in enumerate(ITEM_CATEGORIES):
        if item in ids:
            vec[i] = 1.0
    return vec


def _tera_features(battle, me):
    """Our Terastallization state: can we Tera, our active's Tera type, are we Tera'd.

    All known for our own side (the opponent's Tera type stays hidden until they use it,
    so we don't encode it -- that would just be a blank).
    """
    can_tera = 1.0 if battle.can_tera else 0.0
    if me is not None and me.tera_type is not None:
        tera_type = [1.0 if name == me.tera_type.name else 0.0 for name in TYPE_NAMES]
    else:
        tera_type = [0.0] * len(TYPE_NAMES)
    is_tera = 1.0 if (me is not None and me.is_terastallized) else 0.0
    return [can_tera] + tera_type + [is_tera]


# --- battle-wide feature helpers ---------------------------------------------

def _hazards(side_conditions):
    """Entry hazards on one side: [stealth rock, spikes layers, toxic spikes layers]."""
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
    """Screen-type conditions on one side: [reflect, light screen, aurora veil, tailwind]."""
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
    """One-hot of the active terrain: [electric, grassy, psychic, misty, none]."""
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
    """Whether Trick Room is active (it reverses speed order, so the 'faster' flag flips)."""
    return [1.0 if any(f.name == "TRICK_ROOM" for f in battle.fields) else 0.0]


def _weather(battle):
    """One-hot of the weather: [sun, rain, sand, snow, none]."""
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
    """For up to 5 non-active team members: [hp fraction, offense, defense] vs opponent."""
    opp = battle.opponent_active_pokemon
    feats = []
    benched = [m for m in battle.team.values() if m is not battle.active_pokemon][:5]
    for mon in benched:
        feats += [mon.current_hp_fraction] + _matchup(mon, opp)
    while len(feats) < 15:
        feats.append(0.0)
    return feats


def _opponent_moves(battle):
    """Opponent's revealed moves: up to 4 base powers, plus their effectiveness on us."""
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
    """Opponent set-prediction + damage-estimate features (see knowledge.py).

    All keyed off the opponent's ACTIVE Pokemon (always known) and our own moves, so every
    value here is populated -- no blank features (the lesson from the 854-obs experiment)."""
    # Our moves' estimated damage to the opponent's active (a real "can I KO?" signal).
    my_move_dmg = [0.0] * 4
    for i, move in enumerate(battle.available_moves[:4]):
        my_move_dmg[i] = estimate_damage_fraction(me, move, opp)
    best_out = max(my_move_dmg) if my_move_dmg else 0.0
    # Worst estimated damage the opponent's LIKELY (predicted) moves do to us -> switch cue.
    worst_in = KNOWLEDGE.predicted_incoming(opp, me)

    return (
        my_move_dmg + [best_out, worst_in]      # 6  (worst_in is probability-weighted)
        + KNOWLEDGE.predicted_coverage(opp)     # 18 (P(has attacking move of each type))
        + KNOWLEDGE.role_flags(opp)             # 10 (P(each randbats role))
        + KNOWLEDGE.threat_flags(opp)           # 6  (P of priority/recovery/hazard/setup/status/pivot)
    )


# Total length of the observation vector (see embed_battle for the layout).
# 141 (original) + 4 move priority + 10 own-item flags + 20 Tera = 175,
# + 40 knowledge layer (6 damage + 18 coverage + 10 roles + 6 threats) = 215.
KNOWLEDGE_FEATURES = 6 + 18 + len(ROLE_NAMES) + N_THREAT_FLAGS  # 40
N_FEATURES = 215


def build_observation(battle):
    """Describe the current battle to the network as N_FEATURES numbers.

    Module-level so the same observation can be built outside the env -- the self-play
    opponent (selfplay.py) runs the model on this exact vector inside its choose_move."""
    me = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    # Our active Pokemon's (up to 4) moves.
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
        # Priority decides who moves first regardless of speed (e.g. a priority KO).
        # Clamp to [-1, 1] so a rare -7 move (e.g. Trick Room) can't blow the range.
        # safe_priority: some live pseudo-moves (Struggle) omit the priority field.
        move_priority[i] = max(-1.0, min(1.0, safe_priority(move) / 3.0))

    my_hp = me.current_hp_fraction if me else 0.0
    opp_hp = opp.current_hp_fraction if opp else 0.0
    my_remaining = sum(1 for m in battle.team.values() if not m.fainted) / 6.0
    opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
    opp_remaining = (6 - opp_fainted) / 6.0
    # Effective-speed comparison (accounts for boosts/paralysis), flipped under
    # Trick Room, which reverses the turn order.
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
        + _item_flags(me)                                       # 10 (our held item)
        + _tera_features(battle, me)                            # 20 (Tera: can/type/active)
        + _knowledge_features(battle, me, opp)                  # 40 (set prediction + damage)
    )
    return np.array(features, dtype=np.float32)


# Default reward weights. Winning dominates so the agent plays to win, not just to trade
# evenly; the dense HP/faint/status terms are just enough feedback to guide early learning.
# Self-play raises victory_value mid-curriculum (see train_selfplay.py).
DEFAULT_REWARD = dict(fainted_value=1.0, hp_value=0.5, status_value=0.1, victory_value=100.0)


class ShowdownSinglesEnv(SinglesEnv):
    def __init__(self, reward_weights=None, reward_schedule=None, switch_penalty=0.0, **kwargs):
        super().__init__(**kwargs)
        # Negative low because stat boosts can be negative.
        obs_space = Box(low=-1.0, high=4.0, shape=(N_FEATURES,), dtype=np.float32)
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}
        self.reward_weights = dict(DEFAULT_REWARD, **(reward_weights or {}))

        # Anti-"panic switch" shaping. PokeLLMon found that switching a DIFFERENT Pokemon out
        # on consecutive turns is a top failure mode that correlates with losing -- the agent
        # keeps fleeing matchups instead of committing, burning turns and eating entry-hazard
        # chip. We subtract `switch_penalty` reward whenever the agent makes a voluntary switch
        # on the turn right after another voluntary switch. A single switch is fine (often the
        # right play) and is never penalized; only the SECOND in a row is. 0.0 = disabled.
        self.switch_penalty = switch_penalty
        self._switch_state = {}  # per-battle: {"prev_mon": Pokemon|None, "prev_voluntary": bool}

        # Optional reward ANNEAL. The agent learns to TRADE rather than WIN because the dense
        # per-turn shaping (hp/faint) fires every turn while the victory bonus is rare -- so the
        # consistent gradient rewards trading. Annealing linearly shifts the weights from a
        # `start` (dense shaping, to bootstrap a competent policy) to an `end` (shaping shrunk,
        # so the win bonus dominates and the agent plays to CLOSE games) over `horizon` reward
        # computations. None -> static reward_weights (unchanged v1 behaviour).
        # NOTE horizon is PER-ENV: with N parallel envs, pass total_anneal_steps / N.
        self.reward_schedule = reward_schedule
        if reward_schedule is not None:
            self._anneal_start = dict(DEFAULT_REWARD, **reward_schedule["start"])
            self._anneal_end = dict(DEFAULT_REWARD, **reward_schedule["end"])
            self._anneal_horizon = max(1, int(reward_schedule["horizon"]))
            self._reward_steps = 0

    def embed_battle(self, battle):
        return build_observation(battle)

    def _current_weights(self):
        """The reward weights for this turn: static, or the annealed interpolation if a
        reward_schedule was given (advances one step per call)."""
        if self.reward_schedule is None:
            return self.reward_weights
        frac = min(1.0, self._reward_steps / self._anneal_horizon)
        self._reward_steps += 1
        return {k: self._anneal_start[k] + frac * (self._anneal_end[k] - self._anneal_start[k])
                for k in self._anneal_start}

    def _panic_switch_penalty(self, battle):
        """Penalty for 'panic switching' -- a voluntary switch on the turn right after another
        voluntary switch (see __init__). Returns 0.0 unless switch_penalty is set and this turn
        completes a two-in-a-row voluntary-switch chain.

        Detection: the active Pokemon changed identity since last turn (a switch), and the
        Pokemon we left is still alive (so we CHOSE to switch -- a fainted active forces a switch,
        which is not a panic decision and never counts). A non-switch turn in between resets the
        chain, so ordinary defensive switching is never punished."""
        if not self.switch_penalty:
            return 0.0
        state = self._switch_state.setdefault(battle, {"prev_mon": None, "prev_voluntary": False})
        cur, prev = battle.active_pokemon, state["prev_mon"]
        # Same team slot keeps the same Pokemon object across turns, so identity == "same mon".
        switched = prev is not None and cur is not None and cur is not prev
        voluntary = switched and not prev.fainted
        panic = voluntary and state["prev_voluntary"]
        state["prev_mon"], state["prev_voluntary"] = cur, voluntary
        return self.switch_penalty if panic else 0.0

    def calc_reward(self, battle):
        """Reward = change in how good our position is since last turn (see DEFAULT_REWARD),
        minus an optional panic-switch penalty. With a reward_schedule, the shaping weights
        anneal from start to end across training."""
        reward = self.reward_computing_helper(battle, **self._current_weights())
        return reward - self._panic_switch_penalty(battle)
