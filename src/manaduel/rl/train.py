"""Train a PPO agent against the hand-written bots, and optionally against copies of itself.

    python -m manaduel.rl.train --steps 600000
    python -m manaduel.rl.train --config configs/cards_extended.yaml --memory --selfplay

Every --eval-every steps it plays games against each bot and prints the win
rate, so you can watch it learn. The finished model is saved to --out.

--memory   adds the opponent's played-card counts to what the agent sees
--selfplay after every evaluation, a frozen copy of the current agent joins the pool of
           opponents (the last --keep copies are kept). The agent then trains against
           the four bots AND its own recent past selves, chosen at random each game.
--config   which card set to play (default configs/cards.yaml)
In deck mode (a config with deck_size) each game gives both players a random deck
from the four archetypes.
"""

import argparse
import csv
import time
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from manaduel.agents.registry import BOT_FACTORIES
from manaduel.agents.rl_agent import PPOAgent
from manaduel.engine.decks import archetype_decks
from manaduel.engine.rules import load_game_config
from manaduel.rl.env import ManaDuelEnv
from manaduel.sim.tournament import play_match


def mask_fn(env):
    return env.unwrapped.action_masks()


def make_env(config, memory=False, decks=None):
    def _init():
        env = Monitor(ManaDuelEnv(config, memory=memory, decks=decks))
        return ActionMasker(env, mask_fn)  # must be the outermost wrapper

    return _init


def evaluate(model, config, n_games: int) -> dict:
    """Win rate (draw = half) of the model against each bot, on a fixed set of games."""
    agent = lambda seed: PPOAgent(model)
    return {
        name: play_match(config, "ppo", agent, name, factory, n_games, seed=12345).score_a
        for name, factory in BOT_FACTORIES.items()
    }


def add_snapshot(vec_env, model_path: Path, name: str, memory: bool) -> None:
    """Freeze a saved model and make it available as an opponent in every game environment."""
    frozen = MaskablePPO.load(model_path, device="cpu")
    factory = lambda seed, m=frozen: PPOAgent(m, memory=memory, deterministic=False)
    for env in vec_env.envs:           # DummyVecEnv keeps its environments in this process
        env.unwrapped.add_opponent(name, factory)


def remove_snapshot(vec_env, name: str) -> None:
    for env in vec_env.envs:
        env.unwrapped.remove_opponent(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300_000, help="total training steps")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-games", type=int, default=100, help="games per bot per evaluation")
    parser.add_argument("--envs", type=int, default=8, help="games played in parallel")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoints/ppo_v1")
    parser.add_argument("--log", default="runs/progress.csv", help="where to write the learning curve CSV")
    parser.add_argument("--config", help="card/rule file (default configs/cards.yaml)")
    parser.add_argument("--memory", action="store_true", help="observe the opponent's played cards")
    parser.add_argument("--selfplay", action="store_true", help="also train against frozen copies of itself")
    parser.add_argument("--keep", type=int, default=4, help="how many frozen copies to keep (with --selfplay)")
    args = parser.parse_args()

    config = load_game_config(Path(args.config)) if args.config else load_game_config()
    decks = list(archetype_decks(config).values()) if config.uses_decks else None
    vec_env = DummyVecEnv([make_env(config, args.memory, decks) for _ in range(args.envs)])
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=256,          # steps collected per game before each update
        batch_size=256,
        n_epochs=10,
        gamma=0.99,           # how much future reward matters
        gae_lambda=0.95,
        ent_coef=0.01,        # a little extra exploration
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=args.seed,
        device="cpu",
        verbose=0,
    )

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(BOT_FACTORIES)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshots: list[str] = []
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["steps", *names, "mean"])

        done = 0
        started = time.time()
        while done < args.steps:
            chunk = min(args.eval_every, args.steps - done)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            done += chunk
            scores = evaluate(model, config, args.eval_games)
            mean = sum(scores.values()) / len(scores)
            writer.writerow([done, *[f"{scores[n]:.3f}" for n in names], f"{mean:.3f}"])
            f.flush()
            cells = " | ".join(f"{n} {scores[n]:5.1%}" for n in names)
            note = ""
            if args.selfplay and done < args.steps:
                name = f"self{done // 1000}k"
                path = out.parent / f"{out.name}_{name}.zip"
                model.save(path)
                add_snapshot(vec_env, path, name, args.memory)
                snapshots.append(name)
                if len(snapshots) > args.keep:
                    remove_snapshot(vec_env, snapshots.pop(0))
                note = f" | pool: {len(BOT_FACTORIES)} bots + {len(snapshots)} past selves"
            print(f"{done:>8} steps | {cells} | mean {mean:5.1%} | {time.time() - started:4.0f}s{note}", flush=True)

    model.save(out)
    print(f"Saved model to {out}.zip")


if __name__ == "__main__":
    main()
