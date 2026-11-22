from manaduel.engine.cards import load_config


def test_all_five_cards_load():
    cards, _ = load_config()
    assert set(cards) == {"strike", "fireball", "heal", "shield", "drain"}


def test_fireball_stats():
    cards, _ = load_config()
    fb = cards["fireball"]
    assert fb.cost == 3
    assert fb.damage == 7
    assert fb.heal == 0


def test_drain_does_two_things():
    cards, _ = load_config()
    d = cards["drain"]
    assert d.damage == 3 and d.heal == 3


def test_rules_load():
    _, rules = load_config()
    assert rules.max_hp == 20
    assert rules.max_mana == 5
    assert rules.max_turns == 40
