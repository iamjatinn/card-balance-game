# Card Balance Game

A turn-based card duel, built from scratch, plus the tools to study it: scripted bots, learning agents (PPO), a tree-search player, and a statistical tool that estimates which cards are too strong or too weak.

![Two bots playing a saved replay](docs/demo.gif)

*The original game: a trained PPO agent (left) beats the greedy bot on turn 21.*

![The extended card set](docs/demo_extended.gif)

*The extended card set (draw, poison and combo cards): a trained agent beats the greedy bot on turn 9. Poison counters show under the mana bar.*

## In one minute

**What it is.** A small card game (two players, 20 HP, five to eight cards) and a toolbox around it. The game is the test bench; the point is the toolbox: training an agent with reinforcement learning, measuring card strength without bias, and checking how far those measurements can be trusted.

| Question | Answer | Details |
|---|---|---|
| Can an agent learn the game from win/lose feedback only? | Yes. 48% to 82% win rate against four scripted bots over 28,692 matches (4 minutes on a CPU). Five trained models beat random 94%, aggressive 92%, defensive 71% and greedy 74%. | [story below](#watch-an-agent-learn-match-1-to-match-n), [result 1](docs/EXPERIMENTS.md) |
| Can card strength be measured fairly? | Yes, using the random starting hand as a natural experiment. It found 88% of planted changes (no false alarms in 20 no-change tests) and agrees with itself across halves of 896,877 real Magic games (correlation 0.92). | results 2 to 4 |
| Does "balanced for bots" mean balanced for learned agents? | No. A card set the bots called balanced had one card at +20 points of win probability for trained agents. Training agents inside the balancing loop worked better. | result 10 |
| Why was Shield always weak, and what fixed it? | Only 4 to 12% of its shield ever blocked damage, because it expired too fast, so bigger numbers could not help. A rule change (shield stays until used) fixed it: spread of card power 6.8 to 1.8 points. Side effect: more draws (about 5 to 9%). | result 12 |
| What did not work? | Opponent memory (tested twice), self-play (not shown), longer training (no gain), balancing deck mode. A bigger network helped a little (+1 point, 55.5% head to head). | results 6, 9, 13, 14 |
| Is any of it checked on human players? | No. The tool exists; no human games are logged. | result 11 |

Everything is simulation, and "balanced" always means balanced for the players that were tested. The numbered results are in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md); every number there links to a file in [`docs/results/`](docs/results).

**How the repo is organised.** Read the core first; the extras are experiments built on top of it.

| | What | Where |
|---|---|---|
| Core | the game as a pure function, the bots, PPO training, the card-power estimator and its validation | `engine/`, `agents/`, `rl/`, `analysis/card_effects.py`, `analysis/validate.py` |
| Extras | real-data check, balance search, balancing with agents, decks and metagame, tree search, human-game check, training-story pictures | `analysis/` (the other files), `agents/mcts.py`, `story/` |



## Watch an agent learn (match 1 to match N)

`python -m manaduel.cli story` trains an agent from scratch, records **every training match**, tests the agent at regular checkpoints on the same fresh games, and draws shareable pictures from the saved numbers (nothing is typed in by hand). One real run ([`docs/story/`](docs/story)): 28,692 training matches, 259 seconds of CPU training, no GPU.

![Training story card](docs/story/linkedin_card.png)

| Average win rate vs the four bots | Before training (match 0) | After 28,692 matches |
|---|---|---|
| overall | 48% | **82%** |
| random / aggressive | 59% / 69% | 93% / 93% |
| defensive / greedy | 31% / 34% | 69% / 72% |

What changed in its play, measured on the test games: it plays Shield 2.8x as often (0.9 to 2.4 times per game) and finishes games with an average HP lead of +8.8 instead of -1.5; game length did not change (about 21 turns). Most of the learning happens in the first ~3,000 matches, then the curve flattens at about 80%.

- **Every match as a coloured square** (green won, red lost, grey drew), from match 1 to match 28,692, with a zoom on the first 600 matches where it mostly fails: [`matches_animation.gif`](docs/story/matches_animation.gif). First 500 matches: 46% won; last 500: 85%.
- **One game, five stages**: the same game against the same bot at match 0 (lost) and after about 1k, 5k, 12k and 29k matches (won), with the real win rate of each stage printed on the picture: [`journey.gif`](docs/story/journey.gif). The game is a chosen example (picked so that it agrees with the real win rates); the numbers on the picture are the honest summary.
- Animated curve and win rates growing from match 1: [`learning_animation.gif`](docs/story/learning_animation.gif)
- The untrained and the trained agent in the same game against the same bot: [`before_after.gif`](docs/story/before_after.gif) (a chosen example; the table above is the honest summary)
- Which cards it plays before and after: [`behaviour.png`](docs/story/behaviour.png); a draft caption using the real numbers: [`linkedin_post.md`](docs/story/linkedin_post.md)
- Raw data: [`matches.csv`](docs/story/matches.csv) (one row per training match) and [`checkpoints.csv`](docs/story/checkpoints.csv)

Two curves, not one: the blue line is the results of the matches the agent *trained on* (against a random mix of bots, while still exploring); the green line is the finished policy tested on the same fresh games at each checkpoint, so differences come from the agent alone. The match-0 baseline uses 100 test games per bot and the final point 500, so the starting numbers are noisier. One run, one seed: it shows that learning happened, not how reliably a fresh run lands on 82% (the five-seed results below show a spread of about one point between runs of the same recipe).

```powershell
python -m manaduel.cli story                                  # train + record + draw (about 10 minutes)
python -m manaduel.cli story --steps 100000 --out docs/quick  # a quick version
python -m manaduel.cli story --render-only --footer "your link"   # redraw from an existing recording
```

## The game

Two players start at 20 HP and alternate turns.

- **Mana:** at the start of your own turn it refills to `min(your turn number, 5)`. Unspent mana is lost.
- **Hand:** you start with 3 random cards and draw 1 at the start of each turn (the first player skips the draw on turn 1). The hand limit is 6; extra cards are burned. Draws are random with replacement, weighted by each card's `weight`.
- **On your turn** you may play any number of held, affordable cards, then end the turn.
- **Shield** absorbs damage before HP, is capped at 6, and expires at the start of its owner's next turn (a rule, `shield_keep_percent`, can keep it longer; see result 12).
- You win by reducing the opponent to 0 HP. If both players are alive after turn 40, it is a draw.

| Card | Cost | Effect |
|---|---|---|
| Strike | 1 | Deal 3 damage |
| Fireball | 3 | Deal 7 damage |
| Heal | 2 | Restore 4 HP (max 20) |
| Shield | 1 | Gain 3 shield |
| Drain | 3 | Deal 3 damage and restore 3 HP |

**Three ways to deal the cards.** The rules above are the same everywhere; only where the cards come from changes.

| Mode | Config | Where your cards come from |
|---|---|---|
| Original | `cards.yaml` | endless random draws, like a bag that never empties: five card types, and you can draw Fireball five times in a row |
| Extended | `cards_extended.yaml` (balanced version: `cards_extended_final.yaml`) | the same endless draws, with three more card types |
| Deck mode | `cards_decks.yaml` | each player brings a real 20-card deck and draws from it without replacement, so you choose your cards and the pile can run out |

**Extended set** ([`configs/cards_extended.yaml`](configs/cards_extended.yaml)) adds three cards that use new mechanics: **Insight** (cost 2, draw 2 cards), **Venom** (cost 2, add 3 poison: at the start of the victim's turn they lose HP equal to their poison, ignoring shield, and it drops by 1), **Flurry** (cost 1, 2 damage, +3 if another card was already played this turn).

**Deck mode** ([`configs/cards_decks.yaml`](configs/cards_decks.yaml)): each player brings a finite 20-card deck, drawn without replacement; drawing from an empty deck costs 1, then 2, then 3... HP (fatigue). An optional rule, `deck_max_copies`, limits the copies of one card in a deck (default: no rule).

All stats and rules live in YAML files, so a balance change is a one-line edit. The new mechanics are off by default: **200 recorded v0.1 games are replayed in a test and must stay identical** (they do).

## Results

The full experiments (numbered results 1 to 15, with tables, files and caveats) are in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).
The short version is above; the raw outputs are in [`docs/results/`](docs/results).

## Run it

Requires Python 3.10+. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

(`torch` is only needed for training and the PPO tests, which skip without it.)

```powershell
# play and watch games (add --config configs/cards_extended.yaml for the new cards)
python -m manaduel.cli watch --a greedy --b defensive --seed 8 --save
python -m manaduel.cli view replays/greedy_vs_defensive_seed8.json          # window
python -m manaduel.cli view replays/greedy_vs_defensive_seed8.json --gif demo.gif

# tournaments, training, evaluation
python -m manaduel.cli tournament --n 1000
python -m manaduel.rl.train --config configs/cards_extended.yaml --memory --selfplay --seed 0 --out checkpoints/ext_seed0
python -m manaduel.cli --config configs/cards_extended.yaml multiseed "checkpoints/ext_seed?.zip" --n 500
python -m manaduel.cli --config configs/cards_extended.yaml headtohead "checkpoints/a_seed?.zip" "checkpoints/b_seed?.zip"

python -m manaduel.rl.train --steps 2400000 --net 256 256 --n-steps 512 --batch-size 512 --seed 0 --out checkpoints/big_seed0   # the bigger variant (result 14)

# balance tools
python -m manaduel.cli effects --bot greedy --n 20000
python -m manaduel.cli sweep --set rules.max_shield=3,4,5,6 --bots greedy defensive --n 2000
python -m manaduel.cli validate --bot greedy --reps 2 --null-reps 20
python -m manaduel.cli --config configs/cards_fine.yaml tune --out configs/my_tuned.yaml
python -m manaduel.cli --config configs/cards_decks.yaml metagame --n 2000 --build

# search baseline and trained agents (results 8 and 9)
python -m manaduel.cli mctseval --iterations 25 100 400 --rollout random greedy --opponents random greedy --n 100
python -m manaduel.cli --config configs/cards_decks.yaml multiseed "checkpoints/deck_mem_seed?.zip" --n 500

# balancing with learned agents in the loop (result 10), and your own games against the bots (result 11)
python -m manaduel.analysis.agent_rounds --start configs/cards_extended_balanced.yaml --rounds 4 --out runs/agent_balance
python scripts/shield_diagnosis.py configs/cards.yaml checkpoints/ppo_seed0.zip 300   # how much shield is wasted (result 12)
python -m manaduel.analysis.agent_rounds --start configs/cards_decks_final.yaml --rounds 3 --min-effect 0.015 --gain 0.10 --metagame-n 500 --out runs/deck_balance   # deck mode (result 13)
python -m manaduel.cli play --bot greedy
python -m manaduel.cli humancheck --bot greedy

# real data (download a 17Lands game_data file yourself; CC BY 4.0)
python -m manaduel.cli realdata game_data_public.DSK.PremierDraft.csv.gz --out docs/results/real_data_DSK
```

Use `.\.venv\Scripts\python.exe` in place of `python` if the environment is not activated. Trained models, replays and the 17Lands file are not committed.

## How it is built

```
configs/                  card sets: cards.yaml (original), cards_extended.yaml (+ draw, poison, combo),
                          cards_decks.yaml (deck mode), cards_fine.yaml / cards_fine_balanced.yaml (2x resolution,
                          tuned), cards_extended_fine / _balanced / _agent_balanced / _final.yaml (the extended
                          set through the tuning stages; _final is the balanced one), cards_decks_final.yaml
                          (deck mode with the final cards; _cap4 / _cap3 add a copy limit)
src/manaduel/
  engine/                 state, rules (legal actions, step), game loop, decks
  agents/                 Agent interface, bots, PPO wrapper, MCTS (hidden-hand tree search)
  sim/                    tournaments, Wilson intervals, Elo, JSON replays
  rl/                     observation encoding, Gymnasium env, PPO training (memory, self-play)
  analysis/               card effects, sweep, validate, autobalance, balance_search, metagame,
                          real_data, multiseed, plots, search_eval, agent_balance + agent_rounds
                          (balancing with agents in the loop), human_check
  story/                  records a training run match by match and draws the shareable pictures
  viewer/                 text viewer and pygame replay viewer (PNG / GIF export)
scripts/                  the exact scripts behind the long runs (shield diagnosis, evaluations, final checks)
tests/                    232 tests, including the 200-game v0.1 regression
docs/                     figures, demo GIFs, raw results
```

- **The game is a pure function.** `step(state, action)` returns a new state; the random-number state lives in the game state, so any game replays exactly from its seed.
- **One `Agent` interface** makes random, heuristic and learned bots interchangeable.
- **Action masking:** `MaskablePPO` gives illegal moves zero probability; the environment raises on an illegal action so a masking bug cannot hide.
- **PPO is index-based**, so a model goes stale when the card set changes (the code refuses a model trained on a different card set). The balance tools therefore screen with the bots, which read card stats from the config.

## Limitations

(Result numbers refer to [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).)

- Everything but result 4 is simulation, and balance for bots is not balance for humans. The human-game check (result 11) is built but has no data, so nothing here was checked against human players of this game.
- Result 4 is real data but only one set and event type; effects are conditional on a card being in a deck its player chose; there is no ground truth for true card strength.
- The estimator measures the value of holding a card at the **start** of the game (with how the tested agent uses it), not every effect over a whole game.
- Validation used one bot, two repeats per change and changes of at least 14%; Shield changes could not be detected.
- Balancing is search over whole-number steps. On the original game it did not produce a balanced game (result 5). On the extended set the bot-based search overfitted the bots (result 10), so the final set was corrected with only three trained agents per round, and some levers are coarse (Venom's poison counter can only jump from 3 to 4, a third bigger), so the last round's worst card keeps moving around.
- The MCTS bot (result 8) supports endless-draw games only, was tested with 100 games per setting, and was still improving with more search when measured.
- Memory and self-play comparisons use three models per group; small differences are within seed noise.
- The final extended set (result 12) is balanced for three learned agents in endless-draw play only: not for the hand-written bots, not for deck mode (the deck builder still finds a Heal + Venom deck that beats the field, and Insight behaves differently there; result 13 tried to fix this and could not), and the first-player edge is now slightly reversed (46%). It needed a rule change for Shield. The original extended set is not balanced; the game was built to exercise the tools, not to be a good game.
- Wilson intervals assume independent games; paired deal seeds make them slightly optimistic.
- No license has been granted yet; all rights are reserved by default.

## Roadmap

- [x] Engine, bots, tournaments, replays, PPO (5 seeds), card-power estimator, validation, viewer
- [x] Real-data check, finer balancing search, new mechanics, decks, metagame, memory, self-play
- [x] A balanced version of the extended card set (apply the tools to it) (results 10 and 12: balanced for learned agents in endless-draw play, with Shield fixed by a rule change; not for the bots or for deck mode)
- [ ] Balance deck mode itself (tried in result 13: per-card balancing with agents did not move the metagame; a copy limit of 3 or 4 helps but removes variety; a double-oracle health measure is still missing)
- [x] Train agents in deck mode, where opponent-card memory should matter more (result 9: it made no difference)
- [ ] Check bot-based verdicts against human players (result 11: `play` + `humancheck` are built; needs a few thousand human games, none logged yet)
- [x] MCTS baseline (non-trivial because of hidden hands) (result 8)
