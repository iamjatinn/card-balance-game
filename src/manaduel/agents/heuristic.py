"""Hand-written bots with simple strategies. They are our yardsticks: a learned
agent is only impressive if it beats these."""

from manaduel.agents.base import Agent


def _best_attack(legal, config, me=None):
    """The highest-damage affordable card (cheapest on ties), or None.
    A combo card counts its bonus once another card has been played this turn."""
    attacks = [
        a for a in legal
        if a != config.end_turn and config.cards[a].damage > 0
    ]
    if not attacks:
        return None
    combo_on = me is not None and me.combo > 0

    def power(a):
        card = config.cards[a]
        return card.damage + (card.combo if combo_on else 0)

    return max(attacks, key=lambda a: (power(a), -config.cards[a].cost))


class AggressiveAgent(Agent):
    """Always attacks with the biggest hit it can afford. Never heals or shields."""

    name = "aggressive"

    def __init__(self, seed=None):
        pass  # no randomness, but the constructor matches the other bots

    def act(self, state, legal_actions, config):
        best = _best_attack(legal_actions, config, state.players[state.current])
        return config.end_turn if best is None else best


class DefensiveAgent(Agent):
    """Heals when hurt, keeps a shield up, and attacks with whatever mana is left."""

    name = "defensive"

    def __init__(self, seed=None):
        pass

    def act(self, state, legal_actions, config):
        me = state.players[state.current]
        cards = [a for a in legal_actions if a != config.end_turn]

        # 1. Heal if we are missing a lot of HP (40% of max; 8 of 20 in the default game,
        #    expressed as a fraction so a rescaled game plays the same way).
        if config.rules.max_hp - me.hp >= 0.4 * config.rules.max_hp:
            heals = [
                a for a in cards
                if config.cards[a].heal > 0 and config.cards[a].damage == 0
            ]
            if heals:
                return max(heals, key=lambda a: config.cards[a].heal)

        # 2. Put up a shield if we have none.
        if me.shield == 0:
            shields = [a for a in cards if config.cards[a].shield > 0]
            if shields:
                return max(
                    shields,
                    key=lambda a: config.cards[a].shield / config.cards[a].cost,
                )

        # 3. Otherwise attack, or stop.
        best = _best_attack(legal_actions, config, me)
        return config.end_turn if best is None else best


def _card_value(card, me, opp, rules) -> float:
    """Rough worth of playing a card right now, in 'points'."""
    damage = card.damage + (card.combo if me.combo > 0 else 0)  # combo bonus needs an earlier card this turn
    absorbed = min(opp.shield, damage)
    if damage - absorbed >= opp.hp:
        return 1000.0  # lethal: always take it
    damage_value = (damage - absorbed) + 0.5 * absorbed       # breaking a shield is worth something
    heal_value = min(card.heal, rules.max_hp - me.hp)         # healing past max HP is wasted
    shield_value = 0.7 * min(card.shield, rules.max_shield - me.shield)  # shield above the cap is wasted
    # Extra mechanics (zero for the original five cards, so their values are unchanged):
    room = rules.max_hand - sum(me.hand)                      # cards drawn into a full hand are burned
    draw_value = 1.5 * max(0, min(card.draw, room))
    poison_value = 0.8 * card.poison * (card.poison + 1) / 2  # the total it will deal over the next turns
    return damage_value + heal_value + shield_value + draw_value + poison_value


class GreedyAgent(Agent):
    """Each time, plays the card with the best value per mana. Stops when
    nothing affordable is worth anything."""

    name = "greedy"

    def __init__(self, seed=None):
        pass

    def act(self, state, legal_actions, config):
        me = state.players[state.current]
        opp = state.players[1 - state.current]
        best_action, best_ratio = config.end_turn, 0.0
        for a in legal_actions:
            if a == config.end_turn:
                continue
            card = config.cards[a]
            ratio = _card_value(card, me, opp, config.rules) / card.cost
            if ratio > best_ratio:
                best_action, best_ratio = a, ratio
        return best_action
