"""
Gymnasium-Wrapper für die Wire Harness Multi-Agent Umgebung.
Kapselt Environment so, dass stable-baselines3 SAC direkt genutzt werden kann.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from environment import Environment
from utils.calculations import calculate_center_vector, calculate_rotations, calc_center_dists


class WireHarnessEnv(gym.Env):
    """
    Gymnasium-kompatibler Wrapper für die Wire Harness Umgebung.

    Observation: STATE_SIZE + 1 (letzter Eintrag = aktueller Target-Index)
    Action:      5 kontinuierliche Werte in [-1, 1] (ein Residual-Korrekturwert pro Mover)
    """

    metadata = {"render_modes": []}

    # env.get_states() gibt 30 Werte zurück; +1 für den Target-Index = STATE_SIZE (31)
    OBS_DIM = config.STATE_SIZE        # 31 (30 Features + 1 Ziel-Index)
    ACT_DIM = config.ACTION_SIZE       # 5 Mover

    def __init__(self, max_steps: int = config.MAX_TRAJECTORY_LENGTH,
                 target_order_mode: str = "sequential"):
        super().__init__()

        self.max_steps = max_steps
        self.target_order_mode = target_order_mode
        self._step_count = 0
        self._env: Environment | None = None

        # Observation- und Action-Space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.ACT_DIM,), dtype=np.float32
        )

        # Interne Zustandsvariablen (werden in reset() gesetzt)
        self._vc = None
        self._clockwise = None
        self._angles = None
        self._init_dists = None
        self._move_start = 0
        self._target_sequence = None
        self._mu_targets = None
        self._target_config_idx = 0
        self._prev_dists = None

    # ------------------------------------------------------------------
    def _build_env(self):
        """Erstellt das MuJoCo-Environment einmalig."""
        if self._env is not None:
            return
        target_sequence = config.TARGET_SEQUENCE
        mu_targets = [config.MOVER_TARGETS[target_sequence[i]] for i in range(5)]

        waypoint = []
        for i in range(config.NUM_MOVERS):
            wps = [
                np.sqrt((mu_targets[j][i][0] - mu_targets[j+1][i][0])**2 +
                        (mu_targets[j][i][1] - mu_targets[j+1][i][1])**2)
                for j in range(4)
            ]
            waypoint.append(wps)

        self._env = Environment(
            xml_path=config.XML_PATH,
            num_agents=config.NUM_MOVERS,
            mu_index=config.BODY_IDS,
            mu_start=config.MU_START,
            mu_joints=config.JOINT_NAMES,
            mu_target=mu_targets[0],
            mu_start_move=config.MU_START_MOVE,
            mu_follow=config.MU_FOLLOW,
            mu_max_dist=config.MAX_DIST,
            simend=config.SIMEND,
            table_size=config.TABLE_SIZE,
            vel=config.VEL,
            start_sequence=config.START_SEQUENCE,
            cable_start=config.CABLE_START,
            cable_connect=config.CABLE_CONNECT,
            neighbor=config.NEIGHBOR,
            mu_target2=mu_targets[1],
            mu_target3=mu_targets[2],
            mu_target4=mu_targets[3],
            mu_target5=mu_targets[4],
            waypoint=waypoint,
            cable_end=config.CABLE_END,
            cable_start_mu=config.CABLE_START_MU,
            online_visualizer=False,
        )
        self._env.sim_config_body()

    # ------------------------------------------------------------------
    def _compute_episode_params(self, mu_targets):
        """Berechnet geometrische Größen für eine Episode."""
        vc, cs, ct = calculate_center_vector(config.MU_START,
                                             config.MOVER_TARGETS[self._target_sequence[0]])
        angles = calculate_rotations(config.MU_START,
                                     config.MOVER_TARGETS[self._target_sequence[0]], vc, ct)
        init_dists = calc_center_dists(config.MU_START, cs)
        clockwise = not all(x > 0 for x in angles)
        return vc, clockwise, angles, init_dists

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._build_env()

        if self.target_order_mode == "random":
            self._target_sequence = random.sample(config.TARGET_SEQUENCE,
                                                  len(config.TARGET_SEQUENCE))
        else:
            self._target_sequence = list(config.TARGET_SEQUENCE)

        self._mu_targets = [config.MOVER_TARGETS[self._target_sequence[i]] for i in range(5)]
        self._target_config_idx = self._target_sequence[0]

        # Ziele im Environment setzen
        for i in range(config.NUM_MOVERS):
            self._env.movers[i].set_target(
                self._mu_targets[0][i][0],
                self._mu_targets[0][i][1],
            )

        raw_state = self._env.reset()
        self._move_start = self._env.sim_step
        self._step_count = 0

        self._vc, self._clockwise, self._angles, self._init_dists = \
            self._compute_episode_params(self._mu_targets)

        self._prev_dists = [
            abs(self._env.movers[i].get_distance_target(False))
            for i in range(config.NUM_MOVERS)
        ]

        obs = self._make_obs(raw_state)
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        action_list = np.asarray(action, dtype=np.float32)
        self._step_count += 1

        next_state, reward, done, stop, learn, *_ = self._env.step(
            vc=self._vc,
            clockwise=self._clockwise,
            angles_t=self._angles,
            action_list=action_list,
            init_dists=self._init_dists,
            move_start=self._move_start,
            online_visualizer=False,
        )

        # Dichter Progress-Reward (wie in main.py)
        curr_dists = [
            abs(self._env.movers[i].get_distance_target(False))
            for i in range(config.NUM_MOVERS)
        ]
        progress = sum(
            self._prev_dists[i] - curr_dists[i]
            for i in range(config.NUM_MOVERS)
            if not self._env.movers[i].done
        )
        shaped_reward = float(reward) + 5.0 * progress
        self._prev_dists = curr_dists

        truncated = (self._step_count >= self.max_steps) or bool(stop)
        terminated = bool(done)

        obs = self._make_obs(next_state)
        # reached=True wenn alle Mover ihr Ziel erreicht haben (terminated, nicht truncated)
        info = {"learn": learn, "step": self._step_count, "reached": terminated}
        return obs, shaped_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _make_obs(self, state_list):
        obs = np.array(state_list, dtype=np.float32)
        # Ziel-Index anhängen falls noch nicht enthalten
        if len(obs) < self.OBS_DIM:
            obs = np.append(obs, float(self._target_config_idx))
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    def render(self):
        pass

    def close(self):
        pass
