"""A bot made of rules. No learning involved.

It follows two: hit with whatever does the most damage, and when something faints, bring in
whatever matches up best.

This was stage one of the project. It proved the plumbing worked (connect to the server,
read the battle, send a legal move) and then stuck around as both the benchmark and the
sparring partner for everything that came after.
"""

from poke_env.player import Player


class MaxDamagePlayer(Player):
    """Always clicks the biggest attack, and picks sensibly when forced to switch."""

    def _move_score(self, move, attacker, defender):
        """How hard this move hits. Higher is better."""
        power = move.base_power
        stab = 1.5 if move.type in attacker.types else 1.0  # bonus for matching your own type
        effectiveness = defender.damage_multiplier(move) if defender else 1.0
        return power * stab * effectiveness

    def _matchup_score(self, my_pokemon, opponent):
        """How good a switch this would be: reward hitting them hard, punish being fragile."""
        offense = max(
            (opponent.damage_multiplier(t) for t in my_pokemon.types if t is not None),
            default=1.0,
        )
        defense = max(  # how hard their typing hits us, so lower is better
            (my_pokemon.damage_multiplier(t) for t in opponent.types if t is not None),
            default=1.0,
        )
        return offense - defense

    def choose_move(self, battle):
        opponent = battle.opponent_active_pokemon

        if battle.available_moves:  # an ordinary turn
            best_move = max(
                battle.available_moves,
                key=lambda m: self._move_score(m, battle.active_pokemon, opponent),
            )
            return self.create_order(best_move)

        if battle.available_switches:  # something just fainted
            best_switch = max(
                battle.available_switches,
                key=lambda p: self._matchup_score(p, opponent),
            )
            return self.create_order(best_switch)

        return self.choose_random_move(battle)  # nothing sensible available
