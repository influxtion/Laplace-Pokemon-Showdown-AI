"""A rule-based ("heuristic") Pokemon Showdown bot.

There is NO machine learning here. This bot follows hand-written rules:
  * When it can attack, it picks the move that deals the most damage,
    accounting for the move's base power, STAB, and type effectiveness.
  * When it is forced to switch (its Pokemon fainted), it picks the
    benched Pokemon with the best matchup against the opponent.

This is Stage 1 of the project. It validates the whole pipeline (connecting
to the server, reading the battle state, sending legal actions) and later
becomes the BENCHMARK and TRAINING PARTNER for the reinforcement-learning
agent. Only the scoring logic below gets swapped out for a learned model;
everything around it is reused.
"""

from poke_env.player import Player


class MaxDamagePlayer(Player):
    """Picks the highest-scoring attacking move; switches smartly when forced."""

    def _move_score(self, move, attacker, defender):
        """How good is `move` for `attacker` against `defender`? Higher = better."""
        # Base power of the move (a 0-power move like a status move scores low).
        power = move.base_power

        # STAB: Same-Type Attack Bonus. A move matching the attacker's own type
        # hits 1.5x harder in the real game, so we reward it here too.
        stab = 1.5 if move.type in attacker.types else 1.0

        # Type effectiveness vs the defender (e.g. Water vs Fire = 2x, 0.5x, 0x...).
        # damage_multiplier() reads the gen's type chart for us.
        effectiveness = defender.damage_multiplier(move) if defender else 1.0

        return power * stab * effectiveness

    def _matchup_score(self, my_pokemon, opponent):
        """Rough 'how good is this matchup for me?' score, used when choosing a switch.

        Rewards being able to hit the opponent hard, and punishes being weak to
        the opponent's typing.
        """
        # Best damage we could threaten with our typing (offense).
        offense = max(
            (opponent.damage_multiplier(t) for t in my_pokemon.types if t is not None),
            default=1.0,
        )
        # How much the opponent's typing threatens us (defense; lower is better).
        defense = max(
            (my_pokemon.damage_multiplier(t) for t in opponent.types if t is not None),
            default=1.0,
        )
        return offense - defense

    def choose_move(self, battle):
        opponent = battle.opponent_active_pokemon

        # Normal turn: we have attacking moves available.
        if battle.available_moves:
            best_move = max(
                battle.available_moves,
                key=lambda m: self._move_score(m, battle.active_pokemon, opponent),
            )
            return self.create_order(best_move)

        # Forced switch (our active Pokemon fainted): pick the best matchup.
        if battle.available_switches:
            best_switch = max(
                battle.available_switches,
                key=lambda p: self._matchup_score(p, opponent),
            )
            return self.create_order(best_switch)

        # Nothing sensible to do (rare) -> fall back to a random legal action.
        return self.choose_random_move(battle)
