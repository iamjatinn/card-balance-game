import numpy as np
import pytest

pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from manaduel.engine.rules import load_game_config, new_game  # noqa: E402
from manaduel.rl.env import ManaDuelEnv  # noqa: E402
from manaduel.rl.obs import observation, observation_size  # noqa: E402

CFG = load_game_config()


def test_observation_shape_and_range():
    s = new_game(CFG, seed=3)
    for player in (0, 1):
        obs = observation(s, player, CFG)
        assert obs.shape == (observation_size(CFG),)
        assert obs.dtype == np.float32
        assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_observation_is_from_the_players_point_of_view():
    s = new_game(CFG, seed=3)
    o0 = observation(s, 0, CFG)
    o1 = observation(s, 1, CFG)
    assert o0[-1] == 1.0 and o1[-1] == 0.0  # 'am I the first player' flag
    assert o0[0] == 1.0 and o1[0] == 1.0    # both at full HP
    assert o0[2] > 0.0 and o1[2] == 0.0     # only player 0 has mana on turn 1


def test_gymnasium_checker_passes():
    # The checker tries random (often illegal) actions, so use the lenient mode.
    check_env(ManaDuelEnv(strict=False), skip_render_check=True)


def test_reset_is_reproducible():
    a, _ = ManaDuelEnv().reset(seed=11)
    b, _ = ManaDuelEnv().reset(seed=11)
    assert np.array_equal(a, b)


def test_action_mask_shape_and_end_turn_always_allowed():
    env = ManaDuelEnv()
    env.reset(seed=1)
    mask = env.action_masks()
    assert mask.dtype == bool and mask.shape == (CFG.n_actions,)
    assert mask[CFG.end_turn]


def test_illegal_action_raises():
    env = ManaDuelEnv()
    env.reset(seed=1)
    illegal = np.flatnonzero(~env.action_masks())  # early on, 3-mana cards are unaffordable
    assert len(illegal) > 0
    with pytest.raises(ValueError):
        env.step(int(illegal[0]))


def test_agent_plays_both_seats():
    env = ManaDuelEnv()
    seats = {env.reset(seed=s)[1]["learner_side"] for s in range(20)}
    assert seats == {0, 1}


def test_random_masked_play_always_finishes_with_a_valid_reward():
    env = ManaDuelEnv()
    rng = np.random.default_rng(0)
    rewards = []
    for episode in range(100):
        env.reset(seed=episode)
        done, steps = False, 0
        while not done:
            legal = np.flatnonzero(env.action_masks())
            _, reward, done, truncated, _ = env.step(int(rng.choice(legal)))
            steps += 1
            assert steps < 1000
        rewards.append(reward)
    assert set(rewards) <= {-1.0, 0.0, 1.0}
    assert len(set(rewards)) > 1  # not every episode ends the same way
