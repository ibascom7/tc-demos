import gymnasium as gym
import torch
from model import DQN  # extract the DQN class into its own file, or just redefine it

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder="/home/ibascom/research/taylor-couette/demos/cartPoleTutorial/output/videos", episode_trigger=lambda e: True)

n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load("/home/ibascom/research/taylor-couette/demos/cartPoleTutorial/output/cartpole_dqn.pt", map_location=device))
policy_net.eval()

for ep in range(5):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    total_reward = 0
    while not done:
        with torch.no_grad():
            action = policy_net(state).max(1).indices.item()
        state, reward, terminated, truncated, _ = env.step(action)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {ep}: {total_reward}")

env.close()