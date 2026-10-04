import random
import time as _time

import numpy as np

from env import HighwayEnv


class Agent:

    def __init__(self, env: HighwayEnv, discount_factor = 0.99):
        """
        Initialize the agent.

        Args:
            env: The HighwayEnv environment. Students may use the
                 environment to access its parameters and dynamics.

        Tabular Q-learning over the observation (speed, lane, min_dist[0..3]).
        Only the observation sizes are read from env; everything else is
        learned from env.step().
        """

        self.env = env
        self.gamma = float(discount_factor)

        # observation sizes: speed 0..3, lane 0..3, min_dist[i] in 0..4 per lane
        self.n_speed = env.num_speed_states
        self.n_lanes = env.num_lanes
        self.n_dist = env.num_dist_states
        self.n_actions = env.num_actions
        self.n_states = self.n_speed * self.n_lanes * self.n_dist ** self.n_lanes

        # Q-table and update counts, flat state index x action
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.visits = np.zeros((self.n_states, self.n_actions), dtype=np.int64)

        # learning schedule
        self.alpha = 0.1             # constant step size
        self.eps_start = 1.0         # epsilon decays linearly with elapsed time ...
        self.eps_end = 0.01
        self.eps_decay_frac = 0.8    # ... reaching eps_end at this fraction of the budget

        self.rng = random.Random()
        self.stats = {'steps': 0, 'episodes': 0}

    def _index(self, speed, lane, min_dist):
        i = speed * self.n_lanes + lane
        for d in min_dist:
            i = i * self.n_dist + d
        return i

    def learn_policy(self, time):
        """
        Learn a policy for controlling the car.

        Args:
            time: Maximum time in seconds allowed for learning.

        The learned policy should be stored internally and used by
        `get_action()`.

        Returns:
            None
        """
        start = _time.time()
        # basic guard against run.py's SIGALRM; refined in step B5
        deadline = start + 0.85 * time
        decay_time = self.eps_decay_frac * time
        Q, visits, rng = self.Q, self.visits, self.rng
        g, alpha, n_actions = self.gamma, self.alpha, self.n_actions
        index = self._index
        eps = self.eps_start
        steps = 0

        while _time.time() < deadline:
            # a fresh environment per episode, as run.py does for evaluation
            # (reset() would spawn traffic around the previous episode's car)
            env = HighwayEnv()
            i = index(*env.get_state())
            pending = None              # (state, action, reward) still waiting for its update
            self.stats['episodes'] += 1
            while True:
                if steps % 64 == 0:
                    now = _time.time()
                    if now >= deadline:
                        break
                    eps = max(self.eps_end, self.eps_start - (self.eps_start - self.eps_end) * (now - start) / decay_time)
                if rng.random() < eps:
                    a = rng.randrange(n_actions)
                else:
                    a = int(Q[i].argmax())
                s2, r, done = env.step(a)
                steps += 1
                if done:
                    # env.step checks for a collision *before* applying the action, so on
                    # the step that returns the collision this action never ran and the
                    # reward belongs to the previous action. Credit it there (one-step
                    # lookahead, no bootstrap) and leave this (state, action) untouched.
                    # The same rule at the episode-length limit drops one bootstrap per
                    # surviving episode, which is negligible and needs no reward constants.
                    if pending is not None:
                        pi, pa, pr = pending
                        Q[pi, pa] += alpha * (pr + g * r - Q[pi, pa])
                        visits[pi, pa] += 1
                    break
                if pending is not None:
                    pi, pa, pr = pending
                    Q[pi, pa] += alpha * (pr + g * Q[i].max() - Q[pi, pa])
                    visits[pi, pa] += 1
                pending = (i, a, r)
                i = index(*s2)

        self.stats['steps'] += steps

    def get_action(self, speed, lane, min_dist):
        """
        Select an action for the current state.

        Args:
            speed: Current speed of the controlled car.
            lane: Current lane of the controlled car.
            min_dist: A list where min_dist[i] represents the minimum
                distance between the controlled car and another car
                in lane i.

        Returns:
            int: An action from the following set:

                ACTION_INCREASE_SPEED = 0
                ACTION_DECREASE_SPEED = 1
                ACTION_INCREASE_LANE = 2
                ACTION_DECREASE_LANE = 3
                ACTION_NO_OP = 4
        """
        return int(self.Q[self._index(speed, lane, min_dist)].argmax())
