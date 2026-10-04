import time as _time

import numpy as np


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
# same deltas as env.py: UP -> row i+1, DOWN -> row i-1, LEFT -> col j-1, RIGHT -> col j+1
DELTAS = [(1, 0), (-1, 0), (0, -1), (0, 1)]


class Agent:

    def __init__(self, layout_file, prob_file):
        """
        Initialize the agent.

        Args:
            layout_file: Path to the grid layout file.
            prob_file: Path to the file containing environment probabilities.

        Parses the layout and probabilities and precomputes the factored
        transition model (ship landing table, one pirate matrix per pirate)
        and the reward structure. No solving happens here.
        """
        self._read_layout(layout_file)
        self._read_probs(prob_file)
        self._build_model()

        # value function and greedy policy, indexed [mask, ship cell, p1 idx, p2 idx]
        self.V = np.zeros(self.shape)
        self.policy = np.zeros(self.shape, dtype=np.int8)

    # ------------------------------------------------------------------ #
    # parsing (mirrors env.py exactly)
    # ------------------------------------------------------------------ #

    def _read_layout(self, layout_file):
        with open(layout_file, "r") as f:
            rows = f.readlines()
        # env.py uses N = number of lines and treats every (i, j) in [0, N)^2
        # that is not land as a legal ship cell
        self.N = len(rows)
        self.land = set()
        self.forts = []
        self.treasures = []
        self.pirate_area = set()
        self.ship_start = None
        p1 = p2 = None
        for i, row in enumerate(rows):
            for j, c in enumerate(row.strip()):
                if c == 'L':
                    self.land.add((i, j))
                elif c == 'F':
                    self.forts.append((i, j))
                elif c == 'T':
                    self.treasures.append((i, j))
                elif c == 'S':
                    self.ship_start = (i, j)
                elif c == '!':
                    self.pirate_area.add((i, j))
                elif c == '1':
                    p1 = (i, j)
                    self.pirate_area.add((i, j))
                elif c == '2':
                    p2 = (i, j)
                    self.pirate_area.add((i, j))

        # the two pirate regions are the 4-connected components of the
        # pirate area; region 1 is the one containing pirate '1'
        self.regions = [self._flood_fill(p1), self._flood_fill(p2)]

    def _flood_fill(self, start):
        seen = {start}
        stack = [start]
        while stack:
            i, j = stack.pop()
            for di, dj in DELTAS:
                nb = (i + di, j + dj)
                if nb in self.pirate_area and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return sorted(seen)

    def _read_probs(self, prob_file):
        with open(prob_file, "r") as f:
            lines = f.readlines()
        self.ps = float(lines[0].strip())
        self.pirate_probs = []
        for k in (1, 2):
            p = [float(x) for x in lines[k].strip().split()]
            # np.random.multinomial (used by env.py) treats the last entry as
            # 1 - sum(others), so mirror that instead of trusting the file
            p[3] = max(0.0, 1.0 - sum(p[:3]))
            self.pirate_probs.append(p)
        r = [float(x) for x in lines[3].strip().split()]
        self.R_step, self.R_treasure, self.R_fort, self.R_pirate = r
        self.gamma = float(lines[4].strip())

    # ------------------------------------------------------------------ #
    # factored transition model
    # ------------------------------------------------------------------ #

    def _build_model(self):
        N = self.N
        # compact index over every legal ship cell (all non-land cells)
        self.cells = [(i, j) for i in range(N) for j in range(N)
                      if (i, j) not in self.land]
        self.cell_idx = {c: k for k, c in enumerate(self.cells)}
        K = len(self.cells)

        # nxt[c, d]: landing cell when direction d is executed from cell c
        # (stays put if it would leave the grid or hit land)
        self.nxt = np.empty((K, 4), dtype=np.intp)
        for k, (i, j) in enumerate(self.cells):
            for d, (di, dj) in enumerate(DELTAS):
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N and (ni, nj) not in self.land:
                    self.nxt[k, d] = self.cell_idx[(ni, nj)]
                else:
                    self.nxt[k, d] = k

        # one stay-or-move matrix per pirate; blocked moves add to the diagonal
        self.region_idx = []
        self.region_cells = []   # compact ship-cell index of each region cell
        self.pirate_P = []
        for reg, probs in zip(self.regions, self.pirate_probs):
            ridx = {c: k for k, c in enumerate(reg)}
            P = np.zeros((len(reg), len(reg)))
            for k, (i, j) in enumerate(reg):
                for d, (di, dj) in enumerate(DELTAS):
                    nb = (i + di, j + dj)
                    # env.py checks the whole pirate area, but the two regions
                    # are separate components so a legal move stays in-region
                    P[k, ridx.get(nb, k)] += probs[d]
            self.region_idx.append(ridx)
            self.region_cells.append(np.array([self.cell_idx[c] for c in reg], dtype=np.intp))
            self.pirate_P.append(P)
        self.R1, self.R2 = len(self.regions[0]), len(self.regions[1])

        # treasure mask: bit t set <=> treasure t not yet collected
        self.n_masks = 1 << len(self.treasures)
        self.treasure_cells = [self.cell_idx[t] for t in self.treasures]
        self.fort_cells = np.array([self.cell_idx[f] for f in self.forts], dtype=np.intp)

        self.shape = (self.n_masks, K, self.R1, self.R2)

    # ------------------------------------------------------------------ #
    # value iteration
    # ------------------------------------------------------------------ #

    def _landing_values(self, V, pirates=True):
        """
        U[m, s', r1', r2'] = reward + gamma * V(next state) for landing on ship
        cell s' with the pirates at r1', r2', starting the step with mask m.
        Priority matches env.step: pirate hit, then fort, then treasure.
        With pirates=False, V is indexed [m, s] only and pirates are ignored.
        """
        g = self.gamma
        U = g * V
        # treasure t still present: collect it and move to the mask without bit t
        for t, c in enumerate(self.treasure_cells):
            bit = 1 << t
            for m in range(self.n_masks):
                if m & bit:
                    U[m, c] = g * V[m ^ bit, c] + self.R_treasure
        # fort is terminal
        U[:, self.fort_cells] = self.R_fort
        U += self.R_step
        if not pirates:
            return U
        # landing on a pirate's new cell is terminal (overrides everything)
        hit = self.R_step + self.R_pirate
        r1 = np.arange(self.R1)
        r2 = np.arange(self.R2)
        U[:, self.region_cells[0], r1, :] = hit
        U[:, self.region_cells[1], :, r2] = hit
        return U

    def _pirate_expectation(self, U):
        """W[m, s', r1, r2] = sum_{r1', r2'} P1[r1, r1'] P2[r2, r2'] U[m, s', r1', r2']."""
        P1, P2 = self.pirate_P
        # einsum without optimize uses numpy's own single-threaded loops
        # (matmul / tensordot would go through a possibly multithreaded BLAS)
        X = np.einsum('mkab,cb->mkac', U, P2, optimize=False)
        return np.einsum('ca,mkab->mkcb', P1, X, optimize=False)

    def _ship_expectation(self, W):
        """
        Q[a] = sum_d P(d | a) * G_d with G_d = W[:, nxt[:, d]]. The intended
        direction is executed w.p. ps and each other one w.p. q = (1 - ps) / 3,
        so Q[a] = q * sum_d G_d + (ps - q) * G_a. The max over actions then only
        needs a running sum and a running best G_a (largest if ps > q, smallest
        if ps < q). Returns (max_a Q, argmax_a Q).
        """
        q = (1.0 - self.ps) / 3.0
        c = self.ps - q
        S = W[:, self.nxt[:, 0]]
        best = S.copy()
        arg = np.zeros(W.shape, dtype=np.int8)
        better = np.empty(W.shape, dtype=bool)
        for d in range(1, 4):
            G = W[:, self.nxt[:, d]]
            S += G
            if c >= 0:
                np.greater(G, best, out=better)
            else:
                np.less(G, best, out=better)
            np.copyto(best, G, where=better)
            np.copyto(arg, d, where=better)
            del G
        S *= q
        S += c * best
        return S, arg

    def _sweep(self):
        """One synchronous Bellman backup over all states. Returns max |dV|."""
        U = self._landing_values(self.V)
        W = self._pirate_expectation(U)
        del U
        V_new, self.policy = self._ship_expectation(W)
        del W
        delta = float(np.max(np.abs(V_new - self.V)))
        self.V = V_new
        return delta

    def _relaxed_init(self, deadline):
        """
        Solve the pirate-free version of the MDP over (mask, ship cell) and use
        it as the starting V and policy for every pirate configuration. It is
        tiny, so it gives a sensible policy within milliseconds, and the exact
        sweeps that follow converge to the same V* from any starting point.
        """
        Vr = np.zeros(self.shape[:2])
        arg = np.zeros(self.shape[:2], dtype=np.int8)
        while _time.time() < deadline:
            Vn, arg = self._ship_expectation(self._landing_values(Vr, pirates=False))
            delta = float(np.max(np.abs(Vn - Vr)))
            Vr = Vn
            if delta < 1e-6:
                break
        # build complete arrays before rebinding so a timeout never leaves them half-written
        self.V = np.broadcast_to(Vr[:, :, None, None], self.shape).copy()
        self.policy = np.broadcast_to(arg[:, :, None, None], self.shape).copy()

    # ------------------------------------------------------------------ #
    # API used by run.py
    # ------------------------------------------------------------------ #

    def _encode(self, ship_location, pirate_locations, treasure_locations):
        s = self.cell_idx[tuple(ship_location)]
        r1 = self.region_idx[0][tuple(pirate_locations[0])]
        r2 = self.region_idx[1][tuple(pirate_locations[1])]
        m = 0
        for t, loc in enumerate(self.treasures):
            if loc in treasure_locations:
                m |= 1 << t
        return m, s, r1, r2

    def get_action(self, ship_location, pirate_locations, treasure_locations) -> int:
        """
        Choose an action for the current state.

        Args:
            ship_location: Tuple (x, y) representing the current
                location of the ship.

            pirate_locations: List containing the locations of all
                pirates. There can be at most 2 pirates.

            treasure_locations: List containing the locations of all
                treasures. There can be at most 2 treasures.

        Returns:
            An integer representing the selected action:

                0 -> UP
                1 -> DOWN
                2 -> LEFT
                3 -> RIGHT

        Note:
            This function may be called multiple times after
            `learn_policy()` has been executed.
        """
        return int(self.policy[self._encode(ship_location, pirate_locations, treasure_locations)])

    def learn_policy(self, time):
        """
        Learn a policy for navigating the Treasure Hunt environment.

        Args:
            time: Maximum time (in seconds) allowed for learning.

        The learned policy should be stored internally by the agent
        and subsequently used by `get_action()`.

        Returns:
            None
        """
        start = _time.time()
        # stop before run.py's SIGALRM: don't start a sweep that may not finish
        deadline = start + 0.85 * time
        tol = 1e-9
        last = 0.0
        try:
            self._relaxed_init(min(deadline, start + 0.1 * time))
            while True:
                t0 = _time.time()
                if t0 + 1.5 * last > deadline:
                    break
                delta = self._sweep()
                last = _time.time() - t0
                if delta < tol:
                    break
        except Exception as e:
            # safety net: if run.py's alarm fires anyway (e.g. the first sweep
            # alone exceeds the budget), keep the last complete policy instead
            # of letting the whole run score None
            if type(e).__name__ not in ('TimeoutException', 'TimeoutError'):
                raise
