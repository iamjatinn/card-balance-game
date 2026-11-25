"""Independent check of the three versions of the extended card set on NEW deals (seed 424242), with learned agents
and with hand-written bots. Run from the project root."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from manaduel.analysis import agent_balance
from manaduel.analysis.agent_rounds import _effects_job
from manaduel.analysis.card_effects import collect, estimate_effects
from manaduel.agents.registry import ALL_BOT_FACTORIES
from manaduel.engine.rules import load_game_config

SEED = 424242
VERSIONS = {
    "original (cards_extended)": ("configs/cards_extended.yaml", [f"checkpoints/ext_bots_seed{s}.zip" for s in range(3)]),
    "tuned with bots": ("configs/cards_extended_balanced.yaml", [f"checkpoints/extbal_seed{s}.zip" for s in range(3)]),
    "tuned with agents": ("configs/cards_extended_agent_balanced.yaml",
                          [f"runs/agent_balance/round3/agent_seed{s}.zip" for s in range(3)]),
}
BOTS = ("greedy", "cautious", "bold")


def bot_job(job):
    config_path, bot = job
    config = load_game_config(Path(config_path))
    report = estimate_effects(collect(config, ALL_BOT_FACTORIES[bot], 10000, SEED))
    return [e.effect for e in report.effects], report.seat0_baseline


def main():
    out = {}
    with ProcessPoolExecutor(max_workers=10) as pool:
        agent_futures = {(v, i): pool.submit(_effects_job, (m, cfg, 8000, SEED))
                         for v, (cfg, models) in VERSIONS.items() for i, m in enumerate(models)}
        bot_futures = {(v, b): pool.submit(bot_job, (cfg, b)) for v, (cfg, _) in VERSIONS.items() for b in BOTS}
        for v, (cfg, models) in VERSIONS.items():
            names = [c.name for c in load_game_config(Path(cfg)).cards]
            res = [agent_futures[(v, i)].result() for i in range(len(models))]
            eff = np.array([r[0] for r in res])
            first = [r[1] for r in res]
            bots = {b: bot_futures[(v, b)].result() for b in BOTS}
            out[v] = dict(names=names, agent=agent_balance.summarise(eff, names), agent_first=first,
                          agent_effects=eff.tolist(),
                          bots={b: dict(effects=dict(zip(names, bots[b][0])), first=bots[b][1],
                                        rms=float(np.sqrt(np.mean(np.square(bots[b][0]))))) for b in BOTS})
    Path("docs/results/ext_tuning_final.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = [f"Three versions of the extended card set, measured on NEW deals (seed {SEED}); effects in win-probability points per extra starting copy, vs the median card",
             "agents: mean of 3 trained PPO agents per version (8,000 mirror games each); bots: 10,000 mirror games each", ""]
    cards = out["original (cards_extended)"]["names"]
    for v, d in out.items():
        a = d["agent"]
        lines.append(f"== {v}")
        lines.append("  learned agents: " + "  ".join(f"{n} {a['mean_effect'][n] * 100:+.1f}" for n in d["names"]))
        lines.append(f"    RMS {a['rms'] * 100:.1f}, worst {a['worst_card']} {a['mean_effect'][a['worst_card']] * 100:+.1f}, "
                     f"cards off by 3+ points: {len(a['off_cards'])} ({', '.join(a['off_cards']) or 'none'}); "
                     f"first player wins {np.mean(d['agent_first']) * 100:.1f}% with identical hands")
        for b, bd in d["bots"].items():
            lines.append(f"  {b:<8} bot: " + "  ".join(f"{n} {bd['effects'][n] * 100:+.1f}" for n in d["names"]))
            lines.append(f"    RMS {bd['rms'] * 100:.1f}, first player wins {bd['first'] * 100:.1f}%")
        lines.append("")
    text = "\n".join(lines)
    Path("docs/results/ext_tuning_final.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
