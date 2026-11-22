import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

from sb3_contrib import MaskablePPO  # noqa: E402
from sb3_contrib.common.wrappers import ActionMasker  # noqa: E402

from manaduel.agents.random_agent import RandomAgent  # noqa: E402
from manaduel.agents.rl_agent import PPOAgent  # noqa: E402
from manaduel.engine.game import play_game  # noqa: E402
from manaduel.engine.rules import load_game_config  # noqa: E402
from manaduel.rl.env import ManaDuelEnv  # noqa: E402

CFG = load_game_config()


def test_ppo_agent_plays_a_complete_legal_game():
    env = ActionMasker(ManaDuelEnv(), lambda e: e.unwrapped.action_masks())
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=64, device="cpu", seed=0)
    model.learn(128)  # barely trained: we only check that it plays legally
    result = play_game(CFG, (PPOAgent(model), RandomAgent(1)), seed=2)
    assert result.winner in (0, 1, None)
