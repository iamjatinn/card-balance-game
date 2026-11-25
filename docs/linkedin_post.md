# LinkedIn post: final draft

Edit the wording to sound like you. Every number below is in the repo; the table at the end says where. Do not add
claims the README does not make (for example "balanced for humans": nothing here was checked against people).

**Attach first:** [`docs/story/matches_animation.gif`](story/matches_animation.gif) (every training match as a coloured
square). Second choice: [`docs/story/linkedin_card.png`](story/linkedin_card.png). If LinkedIn will not take the GIF,
turn it into a short video.

---

## Post

I built a small card game from scratch, trained an AI to play it with reinforcement learning, and then used AI agents to find out which cards are broken.

The animation shows every one of the 28,692 matches the agent trained on, as a coloured square (green won, red lost). First 500 matches: 46% won. Last 500: 85%. I programmed no strategy; the agent only learns from win or lose, and it trained in about 4 minutes on a laptop CPU.

What I learned, including the parts that did not work:

- A tool I wrote estimates how strong each card is from the random deal, which acts like a natural experiment. I tested it by planting known changes (it found 88%, with no false alarms in 20 no-change tests), then ran the same method on 896,877 real Magic: The Gathering games. Two random halves of the data agree (correlation 0.92).
- Balancing for scripted bots is not balancing for learned agents. A card set the bots called balanced had one card at +20 points of win probability for the trained agents.
- One card (Shield) could not be fixed with numbers: only 4 to 12% of its shield ever absorbed damage. It needed a rule change.
- Giving the agent a memory of the opponent's played cards did not help, in two game modes.
- A tree-search player that cannot see the opponent's hand plays slightly below the trained agents (it wins 45% against them).

Limits: everything is simulation. Nothing is checked against human players yet, and the card set is balanced for three trained agents, not for deck-building play.

Code, results and the negative results: https://github.com/iamjatinn/card-balance-game

#ReinforcementLearning #MachineLearning #Python #PPO #GameBalance

---

## First comment (optional)

How it works, in one line each: a pure-function game engine (any game replays from its seed); PPO with action masking; a card-power estimator that regresses the result on the cards each side was dealt; a balance search; an information-set tree search; and an agent-in-the-loop balancing round (train agents, measure every card, nudge the off ones, repeat).

## Where each number comes from

| Claim | File |
|---|---|
| 28,692 matches; first 500 won 46%, last 500 won 85%; about 4 minutes | `docs/story/matches.csv`, `docs/story/meta.json`, README "Watch an agent learn" |
| 88% of planted changes found, 0 false alarms in 20 | `docs/results/validation_greedy.txt`, README result 3 |
| 896,877 real games, split-half correlation 0.92 | `docs/results/real_data_DSK.txt`, README result 4 |
| a "balanced for bots" set had a card at +20 points for agents | `docs/results/ext_final_four_versions.txt`, README results 10 and 12 |
| Shield: 4 to 12% of shield points absorbed damage | `scripts/shield_diagnosis.py`, README result 12 |
| memory did not help (original extended set and deck mode) | README results 6 and 9, `docs/results/deck_h2h_memory.txt` |
| the tree search wins 45% of 500 games against the trained agents | `docs/results/mcts_headline.txt`, README result 8 |
