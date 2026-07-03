"""A rule-based Showdown bot. No machine learning.

Two rules: attack with the move that deals the most damage (base power, STAB,
type effectiveness), and when forced to switch, bring in the best matchup.

This is stage 1. It checks the whole pipeline works (server connection, reading
the battle, sending legal actions) and later doubles as the benchmark and
training partner for the RL agent, which reuses everything but the scoring.
"""

from poke_env.player import Player


class MaxDamagePlayer(Player):
    """Picks the highest-scoring attacking move; switches smartly when forced."""

    def _move_score(self, move, attacker, defender):
        """Score `move` for `attacker` against `defender`. Higher is better."""
        power = move.base_power
        stab = 1.5 if move.type in attacker.types else 1.0  # same-type attack bonus
        effectiveness = defender.damage_multiplier(move) if defender else 1.0
        return power * stab * effectiveness

    def _matchup_score(self, my_pokemon, opponent):
        """Switch-in matchup score: reward hitting them hard, punish being weak to them."""
        offense = max(
            (opponent.damage_multiplier(t) for t in my_pokemon.types if t is not None),
            default=1.0,
        )
        defense = max(  # how hard their typing hits us; lower is better
            (my_pokemon.damage_multiplier(t) for t in opponent.types if t is not None),
            default=1.0,
        )
        return offense - defense

    def choose_move(self, battle):
        opponent = battle.opponent_active_pokemon

        if battle.available_moves:  # normal turn
            best_move = max(
                battle.available_moves,
                key=lambda m: self._move_score(m, battle.active_pokemon, opponent),
            )
            return self.create_order(best_move)

        if battle.available_switches:  # forced switch, active fainted
            best_switch = max(
                battle.available_switches,
                key=lambda p: self._matchup_score(p, opponent),
            )
            return self.create_order(best_switch)

        return self.choose_random_move(battle)  # nothing legal to pick
