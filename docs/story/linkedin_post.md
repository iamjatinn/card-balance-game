DRAFT: edit before posting. Every number below comes from this run's saved files.

I built a small card game from scratch and trained an AI to play it using reinforcement learning (PPO).

I did not program any strategy. The agent only gets win / lose feedback after each match.

- Match 0 (untrained): wins 48% of games against my four scripted bots (on average)
- After 28,692 matches (about 4 min of training on a CPU, no GPU): 82%, and 72% against the strongest bot (greedy)
- Tested on 500 fresh games per bot, seats swapped, draws count as half

What changed in its play (measured on the test games):
- it finishes games with an average HP lead of +8.8 (untrained: -1.5)
- it plays Shield 2.8x as often (0.9 to 2.4 times per game)

Honest limits: it beats simple scripted bots; it is not a strong general player. The point of the project was to understand the whole RL loop: environment, action masking, reward design, training and evaluation.

#ReinforcementLearning #MachineLearning #Python #PPO #AI