"""Train a DDPG agent on TaylorCouetteMixingEnv.

Adapted from demos/halfCheetah/main_ddpg.py. Differences:
  - Single-feature Dict observation ({"omega": float}) flattened to np.array.
  - Each env.step() invokes pimpleFoam, so timesteps are budgeted carefully:
    no separate eval env, small start_timesteps, modest total_timesteps.
  - Episodes are truncated at max_steps; never terminated. So done_bool=0
    everywhere -> the critic always bootstraps past episode boundaries.
"""

import argparse
import os
import time

import numpy as np
import torch

import DDPG
from taylor_couette_mixing.envs.taylor_couette_mixing import TaylorCouetteMixingEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_ddpg_tc")
CASE_PATH = "taylor_couette_mixing/cases/tc_mixing_case"


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )


def obs_to_state(obs):
    """Dict obs -> flat np.array. Env returns omega as a scalar; wrap to (1,)."""
    return np.array([float(obs["omega"])], dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps_per_ep", type=int, default=60)
    parser.add_argument("--max_timesteps", type=int, default=3_000)
    parser.add_argument("--start_timesteps", type=int, default=120)
    parser.add_argument("--expl_noise", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    seed = args.seed
    max_steps_per_ep = args.max_steps_per_ep
    max_timesteps = args.max_timesteps
    start_timesteps = args.start_timesteps
    expl_noise = args.expl_noise
    batch_size = args.batch_size
    discount = args.discount
    tau = args.tau
    save_every = args.save_every

    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = TaylorCouetteMixingEnv(case_path=CASE_PATH, max_steps=max_steps_per_ep)

    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = 1   # just omega
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])  # 1.0

    policy = DDPG.DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=discount,
        tau=tau,
    )

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    obs, info = env.reset(seed=seed, options={"reset_mode": "hard"})
    state = obs_to_state(obs)

    episode_reward = 0.0
    episode_timesteps = 0
    episode_num = 0
    episode_returns = []

    total_start = time.time()

    for t in range(max_timesteps):
        episode_timesteps += 1

        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(state)
                + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        step_start = time.time()
        next_obs, reward, terminated, truncated, info = env.step(action)
        step_wall = time.time() - step_start

        next_state = obs_to_state(next_obs)
        done = terminated or truncated
        # Truncation is not a real terminal -> always bootstrap.
        done_bool = float(terminated)

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        print(
            f"t={t+1}/{max_timesteps} ep={episode_num} step={info['step_count']} "
            f"a={action[0]:+.3f} omega={next_obs['omega']:+.2f} "
            f"I={info['mixing_index']:.4f} E={info['energy_consumption']:.3e} "
            f"r={reward:+.4f} dt={step_wall:.1f}s"
        )

        if done:
            episode_returns.append(episode_reward)
            print(
                f"--- episode {episode_num} done. "
                f"return={episode_reward:.3f} len={episode_timesteps} ---"
            )
            obs, info = env.reset(seed=seed, options={"reset_mode": "hard"})
            state = obs_to_state(obs)
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % save_every == 0:
            policy.save(os.path.join(RESULTS_DIR, f"ddpg_tc_t{t+1}"))
            np.save(os.path.join(RESULTS_DIR, "episode_returns.npy"),
                    np.array(episode_returns))

    policy.save(os.path.join(RESULTS_DIR, "ddpg_tc_final"))
    np.save(os.path.join(RESULTS_DIR, "episode_returns.npy"),
            np.array(episode_returns))

    total_time = time.time() - total_start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Episodes completed: {episode_num}")
    if episode_returns:
        print(f"Last 10 episode returns mean: {np.mean(episode_returns[-10:]):.3f}")
