import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import time

import TD3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
VIDEOS_DIR = os.path.join(RESULTS_DIR, "videos")


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
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
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


def record_video(policy, env_name, seed, video_dir, name_prefix):
	"""Run one episode with video recording."""
	env = gym.make(env_name, render_mode="rgb_array")
	env = RecordVideo(env, video_dir, episode_trigger=lambda e: e == 0, name_prefix=name_prefix)

	state, _ = env.reset(seed=seed + 200)
	done = False
	total_reward = 0.
	while not done:
		action = policy.select_action(np.array(state))
		state, reward, terminated, truncated, _ = env.step(action)
		done = terminated or truncated
		total_reward += reward
	env.close()
	print(f"Video '{name_prefix}': reward={total_reward:.3f}")
	return total_reward


def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, _ = eval_env.reset(seed=seed + 100)
		done = False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, terminated, truncated, _ = eval_env.step(action)
			done = terminated or truncated
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	env_name = "HalfCheetah-v5"
	num_seeds = 3
	max_timesteps = int(1e6)
	start_timesteps = int(25e3)
	eval_freq = int(5e3)
	expl_noise = 0.1
	batch_size = 256
	discount = 0.99
	tau = 0.005
	policy_noise = 0.2
	noise_clip = 0.5
	policy_freq = 2

	# Video recording schedule: 0%, 25%, 50%, 75% of training
	video_timesteps = [0, max_timesteps // 4, max_timesteps // 2, 3 * max_timesteps // 4]

	os.makedirs(RESULTS_DIR, exist_ok=True)
	os.makedirs(VIDEOS_DIR, exist_ok=True)

	total_start = time.time()

	best_seed = -1
	best_reward = -float("inf")

	for seed in range(num_seeds):
		print(f"\n{'='*50}")
		print(f"Running seed {seed}")
		print(f"{'='*50}\n")

		env = gym.make(env_name)

		torch.manual_seed(seed)
		np.random.seed(seed)

		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0]
		max_action = float(env.action_space.high[0])

		policy = TD3.TD3(
			state_dim=state_dim,
			action_dim=action_dim,
			max_action=max_action,
			discount=discount,
			tau=tau,
			policy_noise=policy_noise * max_action,
			noise_clip=noise_clip * max_action,
			policy_freq=policy_freq,
		)

		replay_buffer = ReplayBuffer(state_dim, action_dim)

		# Video at start (untrained policy) — only for first seed
		if seed == 0:
			record_video(policy, env_name, seed, VIDEOS_DIR, f"seed{seed}_0_start")

		evaluations = [eval_policy(policy, env_name, seed)]
		videos_recorded = set()

		state, _ = env.reset(seed=seed)
		done = False
		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0

		for t in range(max_timesteps):
			episode_timesteps += 1

			if t < start_timesteps:
				action = env.action_space.sample()
			else:
				action = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * expl_noise, size=action_dim)
				).clip(-max_action, max_action)

			next_state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			done_bool = float(terminated) if episode_timesteps < env.spec.max_episode_steps else 0

			replay_buffer.add(state, action, next_state, reward, done_bool)

			state = next_state
			episode_reward += reward

			if t >= start_timesteps:
				policy.train(replay_buffer, batch_size)

			if done:
				print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
				state, _ = env.reset(seed=seed)
				done = False
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1

			if (t + 1) % eval_freq == 0:
				evaluations.append(eval_policy(policy, env_name, seed))
				np.save(os.path.join(RESULTS_DIR, f"TD3_HalfCheetah-v5_{seed}"), evaluations)

			# Record middle videos at 25%, 50%, 75% — only for first seed
			if seed == 0:
				for vt in video_timesteps[1:]:
					if t + 1 == vt and vt not in videos_recorded:
						pct = int(100 * vt / max_timesteps)
						record_video(policy, env_name, seed, VIDEOS_DIR, f"seed{seed}_{pct}pct")
						videos_recorded.add(vt)

		env.close()

		# Track best seed by final eval reward
		final_reward = evaluations[-1]
		print(f"Seed {seed} final eval reward: {final_reward:.3f}")
		if final_reward > best_reward:
			best_reward = final_reward
			best_seed = seed
			# Save best model
			policy.save(os.path.join(RESULTS_DIR, "best_model"))

	# Record video of the best run
	print(f"\n{'='*50}")
	print(f"Best seed: {best_seed} with reward: {best_reward:.3f}")
	print(f"Recording best policy video...")
	print(f"{'='*50}\n")

	env = gym.make(env_name)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])
	env.close()

	best_policy = TD3.TD3(
		state_dim=state_dim,
		action_dim=action_dim,
		max_action=max_action,
	)
	best_policy.load(os.path.join(RESULTS_DIR, "best_model"))
	record_video(best_policy, env_name, best_seed, VIDEOS_DIR, "best_final")

	total_time = time.time() - total_start
	hours, remainder = divmod(total_time, 3600)
	minutes, seconds = divmod(remainder, 60)
	print(f"\nTotal time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
