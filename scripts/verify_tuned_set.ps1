Set-Location "$env:USERPROFILE\OneDrive\Desktop\card-balance-ai"
$py = "$PWD\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "1"
$res = "$PWD\docs\results"
$orig = "configs/cards_extended.yaml"
$bal = "configs/cards_extended_balanced.yaml"

function Start-Job2($name, $argList) {
    Start-Process -FilePath $py -ArgumentList $argList -RedirectStandardOutput "$PWD\runs\$name.out" -RedirectStandardError "$PWD\runs\$name.err" -WindowStyle Hidden -PassThru
}

# phase 1: three PPO models on the tuned set; card power of the ORIGINAL extended set with the three bots (same seed as the tuning check)
$procs = @()
foreach ($s in 0,1,2) {
    $procs += Start-Job2 "extbal_seed$s" @("-m","manaduel.rl.train","--config",$bal,"--steps","600000","--seed","$s","--out","$PWD\checkpoints\extbal_seed$s","--log","$PWD\runs\extbal_seed$s.csv")
}
foreach ($bot in "greedy","cautious","bold") {
    $procs += Start-Job2 "effects_orig_$bot" @("-m","manaduel.cli","--config",$orig,"effects","--bot",$bot,"--n","20000","--seed","99991")
}
$procs | Wait-Process

foreach ($bot in "greedy","cautious","bold") {
    Copy-Item "$PWD\runs\effects_orig_$bot.out" "$res\effects_ext_original_$bot.txt"
}

# phase 2: card power measured with the trained agents (mirror games), original vs tuned
$procs = @()
foreach ($s in 0,1,2) {
    $procs += Start-Job2 "effects_ppo_extorig_$s" @("-m","manaduel.cli","--config",$orig,"effects","--bot","ppo","--ppo","$PWD\checkpoints\ext_bots_seed$s.zip","--n","8000")
    $procs += Start-Job2 "effects_ppo_extbal_$s" @("-m","manaduel.cli","--config",$bal,"effects","--bot","ppo","--ppo","$PWD\checkpoints\extbal_seed$s.zip","--n","8000")
}
$procs | Wait-Process
foreach ($s in 0,1,2) {
    Copy-Item "$PWD\runs\effects_ppo_extorig_$s.out" "$res\effects_ppo_ext_original_seed$s.txt"
    Copy-Item "$PWD\runs\effects_ppo_extbal_$s.out" "$res\effects_ppo_ext_tuned_seed$s.txt"
}
"ALL DONE" | Out-File "$PWD\runs\verify_tuned.done"
