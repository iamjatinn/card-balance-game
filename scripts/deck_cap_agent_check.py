"""Does a limit on copies per card change the deck metagame when a TRAINED AGENT plays (not only the greedy bot)?

Plays the same eight-deck round robin as the per-round check of analysis/agent_rounds.py (four archetype decks and four
fixed random decks, built to respect each limit), with one deck-mode agent trained without any copy limit, on the same
games for every limit. Run from the project root:

    python scripts/deck_cap_agent_check.py MODEL.zip OUT.txt
"""
import sys
from dataclasses import replace
from pathlib import Path

from manaduel.analysis.agent_rounds import metagame_check
from manaduel.engine.rules import GameConfig, load_game_config

def main():
    model, out = sys.argv[1], Path(sys.argv[2])
    base = load_game_config(Path("configs/cards_decks_final.yaml"))
    lines = ["Eight-deck round robin played by one trained agent (500 games per pairing), by deck copy limit", ""]
    for cap in (0, 4, 3):          # 0 = no rule (decks may hold up to 6 copies)
        config = GameConfig(cards=base.cards, rules=replace(base.rules, deck_max_copies=cap))
        meta = metagame_check(config, model, 500, 99991, 9, out.with_name(f"{out.stem}_cap{cap or 'none'}_matrix.txt"))
        lines.append(f"limit {cap or 'none':>4}: best deck {meta['best_deck']} beats the others {meta['best_average']:.1%} on average, "
                     f"equilibrium uses {meta['equilibrium_decks']} of {meta['n_decks']} decks, {meta['cycles']} cycles, "
                     f"dominant decks {meta['dominant']}")
        print(lines[-1], flush=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
