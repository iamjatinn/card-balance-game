import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

from pathlib import Path  # noqa: E402

from sb3_contrib import MaskablePPO  # noqa: E402
from sb3_contrib.common.wrappers import ActionMasker  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from manaduel.agents.rl_agent import PPOAgent  # noqa: E402
from manaduel.analysis.multiseed import final_eval, head_to_head  # noqa: E402
from manaduel.engine.game import play_game  # noqa: E402
from manaduel.engine.rules import load_game_config  # noqa: E402
from manaduel.rl.env import ManaDuelEnv  # noqa: E402
from manaduel.rl.train import add_snapshot, make_env, mask_fn, remove_snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT_PATH = ROOT / "configs" / "cards_extended.yaml"
EXT = load_game_config(EXT_PATH)


def tiny_model(config, memory=False, steps=64, seed=0):
    env = ActionMasker(ManaDuelEnv(config, memory=memory), mask_fn)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=64, device="cpu", seed=seed)
    model.learn(steps)
    return model


def test_the_agent_works_out_whether_its_model_used_memory():
    plain, with_memory = tiny_model(EXT, False), tiny_model(EXT, True)
    from manaduel.engine.rules import legal_actions, new_game

    s = new_game(EXT, 5)
    for model in (plain, with_memory):
        agent = PPOAgent(model)
        assert agent.act(s, legal_actions(s, EXT), EXT) in legal_actions(s, EXT)
    assert PPOAgent(plain)._memory is None
    agent = PPOAgent(with_memory)
    agent.act(s, legal_actions(s, EXT), EXT)
    assert agent._memory is True


def test_sampling_agents_vary_but_stay_legal():
    model = tiny_model(EXT)
    sampler = PPOAgent(model, deterministic=False)
    result = play_game(EXT, (sampler, PPOAgent(model)), seed=1)
    assert result.winner in (0, 1, None)


def test_snapshots_join_and_leave_the_opponent_pool_in_every_environment(tmp_path):
    vec = DummyVecEnv([make_env(EXT, False, None) for _ in range(3)])
    model = tiny_model(EXT)
    path = tmp_path / "snap.zip"
    model.save(path)
    add_snapshot(vec, path, "self1k", memory=False)
    for env in vec.envs:
        assert "self1k" in env.unwrapped.opponents
    # an environment can actually play against the frozen copy
    env = vec.envs[0].unwrapped
    env.opponents = {"self1k": env.opponents["self1k"]}
    obs, info = env.reset(seed=0)
    assert info["opponent"] == "self1k"
    done = False
    rng = np.random.default_rng(0)
    while not done:
        legal = np.flatnonzero(env.action_masks())
        _, reward, done, _, _ = env.step(int(rng.choice(legal)))
    assert reward in (-1.0, 0.0, 1.0)
    remove_snapshot(vec, "self1k")
    assert all("self1k" not in e.unwrapped.opponents for e in vec.envs)


def test_final_eval_and_head_to_head_with_a_chosen_card_set(tmp_path):
    a = tiny_model(EXT, seed=1)
    b = tiny_model(EXT, memory=True, seed=2)
    pa, pb = tmp_path / "a.zip", tmp_path / "b.zip"
    a.save(pa)
    b.save(pb)
    scores = final_eval([str(pa), str(pb)], n_games=4, seed=1, workers=1, config_path=str(EXT_PATH))
    assert scores.shape == (2, 4) and np.all((scores >= 0) & (scores <= 1))
    duel = head_to_head([str(pa)], [str(pb)], n_games=6, seed=1, workers=1, config_path=str(EXT_PATH))
    assert duel.shape == (1, 1) and 0 <= duel[0, 0] <= 1


def test_a_model_for_the_wrong_card_set_is_refused_by_the_evaluator(tmp_path):
    base_model = tiny_model(load_game_config())
    path = tmp_path / "base.zip"
    base_model.save(path)
    with pytest.raises(ValueError, match="different card set"):
        final_eval([str(path)], n_games=2, seed=0, workers=1, config_path=str(EXT_PATH))
