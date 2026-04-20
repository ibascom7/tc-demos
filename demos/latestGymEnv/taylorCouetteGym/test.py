"""Smoke test: random agent on TaylorCouetteMixingEnv.

Exercises both reset modes:
  - Episode 0: hard reset (pristine initial condition)
  - Episode 1: soft reset (continues from previous episode's final state)

Also demonstrates the omega_mean / omega_amplitude setup: the agent
picks omega within [mean - amp, mean + amp] each step, so the mean
is locked and only the modulation is free -- which is the setting
for studying whether time-varying omega beats constant omega.
"""

import numpy as np

from taylor_couette_mixing.envs.taylor_couette_mixing import TaylorCouetteMixingEnv

CASE_PATH       = "taylor_couette_mixing/cases/tc_mixing_case"
OMEGA_MEAN      = 500.0      # RPM -- baseline the agent modulates around
OMEGA_AMPLITUDE = 100.0      # RPM -- max deviation from the mean
MAX_STEPS       = 5          # keep the smoke test short
SEED            = 0


def run_episode(env, ep_idx, rng, reset_mode):
    obs, info = env.reset(seed=SEED + ep_idx, options={"reset_mode": reset_mode})
    print(f"\n=== Episode {ep_idx}  (reset_mode={reset_mode!r}) ===")
    print(f"  reset obs={obs}  info={info}")

    total_reward = 0.0
    for t in range(MAX_STEPS):
        action = rng.uniform(low=-1.0, high=1.0, size=(1,))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        omega_val = obs["omega"] if isinstance(obs, dict) else obs[0]
        omega_scalar = float(np.asarray(omega_val).ravel()[0])
        print(
            f"  step {t:02d}  "
            f"a={action[0]:+.3f}  "
            f"omega={omega_scalar:+7.2f}  "
            f"I={info['mixing_index']:.4f}  "
            f"E_total={info['energy_consumption']:+.4e}  "
            f"r={reward:+.4f}  "
            f"term={terminated}  trunc={truncated}"
        )

        if terminated or truncated:
            print(f"  -> ended at step {t} (terminated={terminated}, truncated={truncated})")
            break

    print(f"  episode return = {total_reward:+.4f}")


def main():
    env = TaylorCouetteMixingEnv(
        CASE_PATH,
        omega_mean=OMEGA_MEAN,
        omega_amplitude=OMEGA_AMPLITUDE,
        max_steps=MAX_STEPS,
    )
    rng = np.random.default_rng(SEED)

    print("action_space      :", env.action_space)
    print("observation_space :", env.observation_space)
    print(f"omega band        : [{OMEGA_MEAN - OMEGA_AMPLITUDE:.0f}, "
          f"{OMEGA_MEAN + OMEGA_AMPLITUDE:.0f}] RPM")

    # Episode 0: true cold start from 0/
    run_episode(env, 0, rng, reset_mode="hard")

    # Episode 1: pick up from the state the last episode left behind
    run_episode(env, 1, rng, reset_mode="soft")

    close = getattr(env, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
