# Card Balance Game

A small turn-based card duel, written from scratch, with hand-coded bots, a tournament runner, and a reinforcement-learning agent (PPO) trained to play it. The long-term goal is an automated **balance-testing tool**: play thousands of games and find which cards are too strong or too weak.

**Status:** engine, bots, tournaments, replays, and a trained PPO agent are done. The balance-analysis tool and a graphical replay viewer are not built yet (see [Roadmap](#roadmap)).

## The game

Two players start at 20 HP and alternate turns.

- **Mana:** at the start of your own turn it refills to `min(your turn number, 5)`. Unspent mana is lost.
- **Hand:** you start with 3 random cards and draw 1 at the start of each turn (the first player skips the draw on turn 1). The hand limit is 6; extra cards are burned. Draws are random with replacement, weighted by each card's `weight`.
- **On your turn** you may play any number of held, affordable cards, then end the turn.
- **Shield** absorbs damage before HP, is capped at 6, and expires at the start of its owner's next turn.
- You win by reducing the opponent to 0 HP. If both players are alive after turn 40, it is a draw.

| Card | Cost | Effect |
|---|---|---|
| Strike | 1 | Deal 3 damage |
| Fireball | 3 | Deal 7 damage |
| Heal | 2 | Restore 4 HP (max 20) |
| Shield | 1 | Gain 3 shield |
| Drain | 3 | Deal 3 damage and restore 3 HP |

All card stats and rules live in [`configs/cards.yaml`](configs/cards.yaml), not in code, so a balance change is a one-line edit.

## Results

All numbers below come from the code in this repo. They are from a **single training run (seed 0)**, so treat them as indicative, not as averages.

### PPO against the hand-written bots

Trained for 600,000 steps (about 7 minutes on a laptop CPU) against a random mix of the four bots, playing both seats. Then evaluated on fresh games, 500 per pairing (seed 777):

| PPO win rate vs | random | aggressive | defensive | greedy |
|---|---|---|---|---|
| | 93.0% | 92.2% | 69.8% | 73.2% |

Elo from the same tournament: **ppo 1240**, defensive 1091, greedy 1068, random 828, aggressive 773.

During training the average win rate rose from 71.5% (50k steps) to about 80% by 350k steps, then plateaued (79.9% at 600k).

### What the tournaments revealed about the game

- **Pure aggression is a bad strategy.** The "aggressive" bot only plays damage cards, so Heal and Shield cards pile up unused, the hand fills, and new damage cards get burned. It ends up below random in Elo.
- **First-mover advantage had to be tuned.** In the first, deterministic version of the game the first player won almost every game. After adding random draws, a sweep of the second-player compensation showed 1 extra card gave the first mover only 32-35% (smart bots), 0 gave 41-43%, and -1 gave 59-60%. The game ships with 0. A small second-player edge (first mover wins about 41-45% against its own copy for the smart bots) remains.
- **Games can stall.** Across all matchups, 11.7% of games were draws (876 of 7,500). Draws concentrate where both sides heal and shield: defensive vs defensive was 28%. Sustain (Heal, Drain, Shield) is strong relative to damage. This is a natural first target for the balance tool.
- **Deterministic bots in a deterministic game give fake precision.** 1,000 games between two heuristic bots were really two distinct games repeated. Random card draws fixed this, and games now come in seat-swapped pairs that share one draw seed.

## Run it

Requires Python 3.10+. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Run the tests (59, including one that trains a tiny PPO model):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Watch one game, optionally saving a JSON replay to `replays/`:

```powershell
.\.venv\Scripts\python.exe -m manaduel.cli watch --a greedy --b defensive --seed 8 --save
```

Round-robin tournament with win-rate matrix, confidence intervals, Elo, and first-player advantage:

```powershell
.\.venv\Scripts\python.exe -m manaduel.cli tournament --n 1000
```

Train the PPO agent (prints win rates against each bot every 50,000 steps, writes `runs/progress.csv`, saves `checkpoints/ppo_v1.zip`):

```powershell
.\.venv\Scripts\python.exe -m manaduel.rl.train --steps 600000 --eval-every 50000 --eval-games 200
```

Include the trained agent in a tournament:

```powershell
.\.venv\Scripts\python.exe -m manaduel.cli tournament --n 500 --ppo checkpoints/ppo_v1.zip
```

The trained model is not committed; train your own with the command above.

## How it is built

```
configs/cards.yaml     card stats and rules
src/manaduel/
  engine/              state, rules (legal actions, step), game loop
  agents/              Agent interface, random + heuristic bots, PPO wrapper
  sim/                 tournaments, Wilson intervals, Elo, JSON replays
  rl/                  observation encoding, Gymnasium env, PPO training
  viewer/              text viewer for finished games
tests/                 59 tests
```

Design choices worth knowing:

- **The game is a pure function.** `step(state, action)` returns a new state and never modifies the old one. The random-number-generator state lives inside the game state, so any game replays exactly from its seed.
- **One `Agent` interface** (`act(state, legal_actions, config)`) means random, heuristic, and learned bots are interchangeable in tournaments.
- **Action masking.** The RL environment exposes which actions are legal, and `MaskablePPO` assigns illegal moves zero probability. The environment raises an error on an illegal action, so a masking bug cannot hide.
- **Sparse reward.** +1 for a win, -1 for a loss, 0 for a draw, nothing otherwise.
- **Statistics.** Win rates carry 95% Wilson intervals, and draws count as half a win.

## Limitations

- One training run and one seed. The PPO numbers have not been averaged over several runs.
- PPO was trained and tested against four simple, fixed bots. Beating them does not mean it plays near-optimally, and it was not tested against strong human play or a copy of itself.
- The Wilson intervals assume independent games. Paired seeds make them slightly optimistic.
- Everything is simulation. Bot balance is not the same as human balance.
- The model sees only the opponent's hand *size*, not their cards.

## Roadmap

- [ ] Balance-analysis tool: per-card statistics, planted-exploit detection, and a parameter sweep command
- [ ] Graphical replay viewer (pygame)
- [ ] Self-play training and multiple training seeds
- [ ] MCTS baseline agent
