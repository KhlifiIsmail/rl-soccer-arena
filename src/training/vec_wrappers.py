from __future__ import annotations

from stable_baselines3.common.vec_env import VecEnvWrapper


class CleanEpisodeInfoVecWrapper(VecEnvWrapper):
    """
    SB3 expects info["episode"] to be a dict like {"r": float, "l": int, "t": float}.
    If any env writes info["episode"] = int/float, SB3 can crash in dump_logs().

    This wrapper:
      - detects invalid info["episode"] values (non-dict)
      - moves them to another key (default: "episode_scalar")
      - deletes the broken "episode" key before SB3 sees it
    """

    def __init__(self, venv, rename_to: str = "episode_scalar"):
        super().__init__(venv)
        self.rename_to = rename_to

    def reset(self, **kwargs):
        # Just forward reset to the wrapped VecEnv
        return self.venv.reset(**kwargs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        for info in infos:
            if not isinstance(info, dict):
                continue

            if "episode" in info and not isinstance(info["episode"], dict):
                info[self.rename_to] = info["episode"]
                del info["episode"]

        return obs, rewards, dones, infos
