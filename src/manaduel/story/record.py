"""Record a training run match by match.

What gets saved in the output folder:
  matches.csv      one row per TRAINING match: match number, result (1 win / 0.5 draw / 0 loss),
                   opponent, which seat the agent had, and how many turns it lasted
  checkpoints.csv  one row per checkpoint (match 0 = before any training): the agent's win rate
                   against each bot on a FIXED set of fresh games, the average game length, and
                   how many times per game it plays each card
  models/          the model saved at every checkpoint (ckpt_<steps>.zip)
  meta.json        what was run, and how long training took (`seconds` = training only,
                   `wall_seconds` = including the checkpoint tests)

Two different win-rate curves come out of this, and they are not the same thing:
  * the training curve (matches.csv): results of the matches the agent actually trained on,
    against a random mix of bots, while it was still exploring;
  * the evaluation curve (checkpoints.csv): the finished policy at that moment (no exploration),
    against each bot, on the same games every time, so differences come from the agent alone.
"""

import csv
import json
import time
from collections import Counter
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from manaduel.agents.registry import BOT_FACTORIES
from manaduel.agents.rl_agent import PPOAgent
from manaduel.engine.game import play_game
from manaduel.rl.train import build_model, make_env

MATCH_COLUMNS = ["match", "step", "result", "opponent", "seat", "turns"]


class MatchLogger(BaseCallback):
    """Notes the result of every finished training match."""

    def __init__(self):
        super().__init__()
        self.matches = 0
        self.rows: list[list] = []

    def _on_step(self) -> bool:
        dones, rewards, infos = self.locals["dones"], self.locals["rewards"], self.locals["infos"]
        for i, done in enumerate(dones):
            if not done:
                continue
            self.matches += 1
            reward = float(rewards[i])
            result = 1.0 if reward > 0 else 0.0 if reward < 0 else 0.5
            info = infos[i]
            self.rows.append([self.matches, self.num_timesteps, result, info.get("opponent", "?"),
                              info.get("learner_side", -1), info.get("turns", -1)])
        return True

    def take(self) -> list[list]:
        rows, self.rows = self.rows, []
        return rows


def evaluate_model(model, config, n_games: int, seed: int = 12345, bots: dict | None = None) -> dict:
    """Play `n_games` against every bot (seats swapped, same deals every call) with the finished policy.

    Returns the win rate against each bot (draw = half), their mean, the average game length,
    the agent's final HP lead, and how often per game it plays each card."""
    bots = bots or BOT_FACTORIES
    win: dict[str, float] = {}
    plays: Counter = Counter()
    games = turns = 0
    margin = 0.0
    for name, factory in bots.items():
        points = 0.0
        for g in range(n_games):
            agent_first = g % 2 == 0
            agent, bot = PPOAgent(model), factory(seed + 2 * g + 1)
            agents = (agent, bot) if agent_first else (bot, agent)
            result = play_game(config, agents, seed=seed + 1_000_003 * (g // 2 + 1))
            seat = 0 if agent_first else 1
            if result.winner is None:
                points += 0.5
            elif result.winner == seat:
                points += 1.0
            for event in result.events:
                if event["player"] == seat and event["action"] != "end_turn":
                    plays[event["action"]] += 1
            last = result.events[-1]
            margin += last["hp"][seat] - last["hp"][1 - seat]
            turns += result.turns
            games += 1
        win[name] = points / n_games
    return {
        "win": win,
        "mean": sum(win.values()) / len(win),
        "turns": turns / games,
        "hp_margin": margin / games,
        "plays_per_game": {card.name: plays[card.name] / games for card in config.cards},
        "games_per_bot": n_games,
    }


def _checkpoint_row(matches, steps, seconds, ev, config):
    return ([matches, steps, round(seconds, 1), ev["games_per_bot"], round(ev["mean"], 4)]
            + [round(ev["win"][b], 4) for b in BOT_FACTORIES]
            + [round(ev["turns"], 2), round(ev["hp_margin"], 2)]
            + [round(ev["plays_per_game"][c.name], 3) for c in config.cards])


def checkpoint_columns(config) -> list[str]:
    return (["matches", "steps", "seconds", "eval_games", "mean"] + list(BOT_FACTORIES)
            + ["turns", "hp_margin"] + [f"plays_{c.name}" for c in config.cards])


def run_story(config, out_dir, steps: int = 600_000, chunk_steps: int = 25_000, eval_games: int = 100,
              final_games: int = 500, envs: int = 8, seed: int = 0, config_name: str = "cards.yaml",
              log=print) -> Path:
    """Train from scratch for `steps` steps, recording every match and evaluating at each checkpoint."""
    out = Path(out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    vec_env = DummyVecEnv([make_env(config) for _ in range(envs)])
    model = build_model(vec_env, seed)
    logger = MatchLogger()
    started = time.time()

    meta = {"config": config_name, "steps": steps, "chunk_steps": chunk_steps, "eval_games": eval_games,
            "final_games": final_games, "envs": envs, "seed": seed, "algorithm": "MaskablePPO (sb3-contrib)",
            "opponents": list(BOT_FACTORIES), "finished": False}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with open(out / "matches.csv", "w", newline="", encoding="utf-8") as mf, \
            open(out / "checkpoints.csv", "w", newline="", encoding="utf-8") as cf:
        match_writer, check_writer = csv.writer(mf), csv.writer(cf)
        match_writer.writerow(MATCH_COLUMNS)
        check_writer.writerow(checkpoint_columns(config))

        # checkpoint 0: the untrained agent, before it has played a single match
        model.save(out / "models" / "ckpt_00000000.zip")
        first = evaluate_model(model, config, eval_games)
        check_writer.writerow(_checkpoint_row(0, 0, 0.0, first, config))
        cf.flush()
        log(f"match 0 (untrained): mean win rate {first['mean']:.1%}")

        done = 0
        train_time = 0.0                      # seconds spent training (not testing)
        while done < steps:
            chunk = min(chunk_steps, steps - done)
            t0 = time.time()
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=logger)
            train_time += time.time() - t0
            done += chunk
            match_writer.writerows(logger.take())
            mf.flush()
            last_chunk = done >= steps
            ev = evaluate_model(model, config, final_games if last_chunk else eval_games)
            path = out / "models" / f"ckpt_{done:08d}.zip"
            model.save(path)
            check_writer.writerow(_checkpoint_row(logger.matches, done, train_time, ev, config))
            cf.flush()
            cells = " | ".join(f"{b} {ev['win'][b]:5.1%}" for b in BOT_FACTORIES)
            log(f"match {logger.matches:>6,} (step {done:>8,}): {cells} | mean {ev['mean']:5.1%} "
                f"| {time.time() - started:4.0f}s")
        model.save(out / "models" / "final.zip")

    meta.update({"finished": True, "matches": logger.matches, "seconds": round(train_time, 1),
                 "wall_seconds": round(time.time() - started, 1)})
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
