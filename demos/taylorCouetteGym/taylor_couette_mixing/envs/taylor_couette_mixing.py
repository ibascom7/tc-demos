import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from taylor_couette_mixing.envs.helpers import Helpers

# Helpers is finished
# need to write reset and step
# figure out render


class TaylorCouetteMixingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, case_path, render_mode=None):
        self.helpers = Helpers(case_path)

        self.omega = 0
        self.alpha = 1.0 # How much we care about mixing
        self.beta = 1.0 # How much we care about energy consumption
        self.I_current = 1.0 # Initially unmixed
        self.E_current = 0.0 # 0 initial energy consumption
        self.time_step = 5 # Number of seconds that the sim runs between steps

        self.observation_space = spaces.Dict(
            {
                "omega": spaces.Box(low=-1000.0, high=1000.0, shape=(1,), dtype=np.float64)
            }
        )

        # -1 <= delta_omega <= 1 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)


        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        # self.render_mode = render_mode

        # """
        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        # """
        # self.window = None
        # self.clock = None

    def _get_obs(self):
        return {"omega": self.omega}

    def _get_info(self):
        return {
            "mixing_index": self.I_current,
            "energy_consumption": self.E_current
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Starts at 0 rpm
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        delta_omega = 250.0 * action[0]
        self.omega = np.clip(self.omega + delta_omega, -1000.0, 1000.0)
        omega_rad = (self.omega*(2*np.pi))/60

        results = self.helpers.do_simulation(omega_rad, self.time_step)

        # Mz_kin is torque / density given by OpenFOAM
        powers = []
        times = []
        for result in results:
             Mz = result["Mz_kin"] * 1000
             powers.append(Mz * omega_rad)
             times.append(result["t"])
        E = np.trapz(powers, times) # Energy consumption of this time step

        # OpenFOAM simulates a r by h wedge of the annulus
        # There are 20 equally spaced bins between r_in and r_out at the bottom with concentrations at each.
        final_result = results[-1]
        concentrations = [final_result[f"C{i}"] for i in range(20)]
        C_cup = 1

        terminated = (results.I < results.I_threshold)
        reward = 1 if terminated else (-(self.alpha * results.I) - (self.beta * E))
        observation = self._get_obs()

        self.E_current = self.E_current + E
        info = self._get_info()
        return observation, reward, terminated, False, info

    # def render(self):
    #     if self.render_mode == "rgb_array":
    #         return self._render_frame()

    # def _render_frame(self):
    #     if self.window is None and self.render_mode == "human":
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode((self.window_size, self.window_size))
    #     if self.clock is None and self.render_mode == "human":
    #         self.clock = pygame.time.Clock()

    #     canvas = pygame.Surface((self.window_size, self.window_size))
    #     canvas.fill((255, 255, 255))
    #     pix_square_size = (
    #         self.window_size / self.size
    #     )  # The size of a single grid square in pixels

    #     # First we draw the target
    #     pygame.draw.rect(
    #         canvas,
    #         (255, 0, 0),
    #         pygame.Rect(
    #             pix_square_size * self._target_location,
    #             (pix_square_size, pix_square_size),
    #         ),
    #     )
    #     # Now we draw the agent
    #     pygame.draw.circle(
    #         canvas,
    #         (0, 0, 255),
    #         (self._agent_location + 0.5) * pix_square_size,
    #         pix_square_size / 3,
    #     )

    #     # Finally, add some gridlines
    #     for x in range(self.size + 1):
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (0, pix_square_size * x),
    #             (self.window_size, pix_square_size * x),
    #             width=3,
    #         )
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (pix_square_size * x, 0),
    #             (pix_square_size * x, self.window_size),
    #             width=3,
    #         )

    #     if self.render_mode == "human":
    #         # The following line copies our drawings from `canvas` to the visible window
    #         self.window.blit(canvas, canvas.get_rect())
    #         pygame.event.pump()
    #         pygame.display.update()

    #         # We need to ensure that human-rendering occurs at the predefined framerate.
    #         # The following line will automatically add a delay to
    #         # keep the framerate stable.
    #         self.clock.tick(self.metadata["render_fps"])
    #     else:  # rgb_array
    #         return np.transpose(
    #             np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
    #         )

    # def close(self):
    #     if self.window is not None:
    #         pygame.display.quit()
    #         pygame.quit()
