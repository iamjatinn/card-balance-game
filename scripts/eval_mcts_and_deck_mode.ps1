Set-Location "$env:USERPROFILE\OneDrive\Desktop\card-balance-ai"
$py = "$PWD\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$res = "docs\results"
$cfg = "configs/cards_decks.yaml"

# --- deck mode: final evaluation of the six models on the same fresh games (all deck pairings)
& $py -m manaduel.cli --config $cfg multiseed "checkpoints/deck_nomem_seed?.zip" --n 500 --workers 6 --csv "$res\deck_nomem_final.csv" | Tee-Object "$res\deck_nomem_final.txt"
& $py -m manaduel.cli --config $cfg multiseed "checkpoints/deck_mem_seed?.zip" --n 500 --workers 6 --csv "$res\deck_mem_final.csv" | Tee-Object "$res\deck_mem_final.txt"
# --- deck mode: model against model
& $py -m manaduel.cli --config $cfg headtohead "checkpoints/deck_mem_seed?.zip" "checkpoints/deck_nomem_seed?.zip" --label-a "memory" --label-b "no memory" --n 400 --workers 6 | Tee-Object "$res\deck_h2h_memory.txt"
& $py -m manaduel.cli --config $cfg headtohead "checkpoints/deck_nomem_seed?.zip" "checkpoints/deck_nomem_seed?.zip" --label-a "no memory" --label-b "no memory (same group, control)" --n 400 --workers 6 | Tee-Object "$res\deck_h2h_control.txt"
& $py -m manaduel.cli --config $cfg headtohead "checkpoints/deck_mem_seed?.zip" "checkpoints/deck_mem_seed?.zip" --label-a "memory" --label-b "memory (same group, control)" --n 400 --workers 6 | Tee-Object "$res\deck_h2h_control_mem.txt"

# --- MCTS: how strength grows with search effort, and the headline setting against bots and trained agents
& $py -m manaduel.cli mctseval --iterations 25 100 400 --rollout random greedy --opponents random greedy --n 100 --workers 6 --out "$res\mcts_effort.txt"
& $py -m manaduel.cli mctseval --iterations 200 --rollout greedy --opponents random aggressive defensive greedy "ppo:checkpoints/ppo_seed0.zip" "ppo:checkpoints/ppo_seed1.zip" "ppo:checkpoints/ppo_seed2.zip" "ppo:checkpoints/ppo_seed3.zip" "ppo:checkpoints/ppo_seed4.zip" --n 100 --workers 6 --out "$res\mcts_headline.txt"
"ALL DONE" | Out-File runs\eval_all.done
