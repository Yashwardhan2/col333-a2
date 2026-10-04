"""Prototypes of the review suggestions (layered VI, BFS init, relaxed pirate-free init,
band Gauss-Seidel), built on top of the current Agent without modifying it.
Measurement only -- nothing here is used by the submitted agent."""
import sys, os, time
from collections import deque
import numpy as np
PA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'A2-starter-code', 'A2-starter-code', 'part_a')
sys.path.insert(0, PA)
from agent import Agent, DELTAS


class Proto(Agent):

    # ---------- generic pieces restricted to a mask subset / cell subset ----------

    def landing(self, V, masks, cells=None):
        """U for masks `masks` (list) and landing cells `cells` (index array or None = all)."""
        g = self.gamma
        cells_arr = np.arange(self.shape[1]) if cells is None else cells
        pos = {int(c): i for i, c in enumerate(cells_arr)}
        U = g * V[np.ix_(masks, cells_arr)] if cells is not None else g * V[masks]
        for t, c in enumerate(self.treasure_cells):
            if c not in pos: continue
            bit = 1 << t
            for i, m in enumerate(masks):
                if m & bit:
                    U[i, pos[c]] = g * V[m ^ bit, c] + self.R_treasure
        for f in self.fort_cells:
            if int(f) in pos: U[:, pos[int(f)]] = self.R_fort
        U += self.R_step
        hit = self.R_step + self.R_pirate
        for which, rc in enumerate(self.region_cells):
            ks = [k for k, c in enumerate(rc) if int(c) in pos]
            if not ks: continue
            li = np.array([pos[int(rc[k])] for k in ks]); ks = np.array(ks)
            if which == 0: U[:, li, ks, :] = hit
            else: U[:, li, :, ks] = hit
        return U

    def ship_fixed(self, W, nxt_rows, pi):
        """Q under a fixed policy pi (same leading shape as result)."""
        q = (1.0 - self.ps) / 3.0; c = self.ps - q
        S = None; Gp = None
        for d in range(4):
            G = W[:, nxt_rows[:, d]]
            S = G.copy() if S is None else S + G
            Gp = np.where(pi == d, G, 0.0) if Gp is None else Gp + np.where(pi == d, G, 0.0)
        return q * S + c * Gp

    def sweep_masks(self, masks):
        U = self.landing(self.V, masks)
        W = self._pirate_expectation(U)
        Vn, arg = self._ship_expectation(W)
        d = float(np.max(np.abs(Vn - self.V[masks])))
        self.V[masks] = Vn; self.policy[masks] = arg
        return d

    def greedy(self, V):
        U = self.landing(V, list(range(self.n_masks)))
        return self._ship_expectation(self._pirate_expectation(U))

    def evaluate(self, pi, tol=1e-6, max_it=6000):
        V = np.zeros(self.shape); allm = list(range(self.n_masks))
        for _ in range(max_it):
            W = self._pirate_expectation(self.landing(V, allm))
            Vn = self.ship_fixed(W, self.nxt, pi)
            d = np.max(np.abs(Vn - V)); V = Vn
            if d < tol: break
        return V

    def start_index(self, layout):
        from env import TreasureHunt
        return self._encode(*TreasureHunt(layout, layout.replace('_layout', '_prob').replace('layout.txt', 'prob.txt')).get_state())

    # ---------- initialisations ----------

    def bfs_dist(self, targets):
        K = self.shape[1]; dist = np.full(K, np.inf); dq = deque()
        for t in targets: dist[t] = 0; dq.append(t)
        # reverse edges: s -> nxt[s, d]; BFS from targets over predecessors
        preds = [[] for _ in range(K)]
        for s in range(K):
            for d in range(4):
                n = self.nxt[s, d]
                if n != s: preds[n].append(s)
        while dq:
            u = dq.popleft()
            for p in preds[u]:
                if dist[p] == np.inf: dist[p] = dist[u] + 1; dq.append(p)
        return dist

    def init_bfs_gemini(self):
        """Exactly as suggested: m=0 layer = R_step * dist + R_fort, others 0."""
        dist = self.bfs_dist(list(self.fort_cells))
        v = np.where(np.isfinite(dist), self.R_step * dist + self.R_fort, 0.0)
        self.V[0] = v[:, None, None]

    def init_relaxed(self):
        """Solve the pirate-free MDP over (mask, cell) and broadcast it."""
        K = self.shape[1]; g = self.gamma
        Vr = np.zeros((self.n_masks, K))
        q = (1.0 - self.ps) / 3.0; c = self.ps - q
        for _ in range(100000):
            U = g * Vr
            for t, cc in enumerate(self.treasure_cells):
                bit = 1 << t
                for m in range(self.n_masks):
                    if m & bit: U[m, cc] = g * Vr[m ^ bit, cc] + self.R_treasure
            U[:, self.fort_cells] = self.R_fort
            U += self.R_step
            G = np.stack([U[:, self.nxt[:, d]] for d in range(4)])
            Vn = q * G.sum(0) + c * (G.max(0) if c >= 0 else G.min(0))
            dd = np.max(np.abs(Vn - Vr)); Vr = Vn
            if dd < 1e-9: break
        self.V[:] = Vr[:, :, None, None]

    # ---------- band Gauss-Seidel (cells ordered by BFS distance to targets) ----------

    def setup_bands(self):
        targets = list(self.fort_cells) + list(self.treasure_cells)
        dist = self.bfs_dist(targets)
        dist[~np.isfinite(dist)] = dist[np.isfinite(dist)].max() + 1
        self.bands = []
        for b in np.unique(dist):
            C = np.nonzero(dist == b)[0]
            L = np.unique(self.nxt[C].ravel())
            pos = np.searchsorted(L, self.nxt[C])
            self.bands.append((C, L, pos))

    def sweep_gs(self):
        allm = list(range(self.n_masks)); delta = 0.0
        q = (1.0 - self.ps) / 3.0; c = self.ps - q
        for C, L, pos in self.bands:
            W = self._pirate_expectation(self.landing(self.V, allm, L))
            S = W[:, pos[:, 0]]; best = S.copy(); arg = np.zeros(S.shape, np.int8)
            for d in range(1, 4):
                G = W[:, pos[:, d]]; S = S + G
                better = G > best if c >= 0 else G < best
                best = np.where(better, G, best); arg[better] = d
            Vn = q * S + c * best
            delta = max(delta, float(np.max(np.abs(Vn - self.V[:, C]))))
            self.V[:, C] = Vn; self.policy[:, C] = arg
        return delta


def run_variant(layout, prob, variant, tol=1e-9, max_time=120, snap_every=1):
    ag = Proto(layout, prob)
    t_init = time.time()
    if variant in ('relaxed', 'layered+relaxed', 'gs+relaxed'): ag.init_relaxed()
    if variant == 'bfs_gemini': ag.init_bfs_gemini()
    if variant.startswith('gs'): ag.setup_bands()
    t_init = time.time() - t_init
    snaps = [(t_init, ag.V.copy())]
    t = t_init; n = 0
    if variant.startswith('layered'):
        phases = [[0]] + [[m for m in range(1, ag.n_masks - 1)]] + [[ag.n_masks - 1]]
        for ms in phases:
            while True:
                t0 = time.time(); d = ag.sweep_masks(ms); t += time.time() - t0; n += 1
                if n % snap_every == 0: snaps.append((t, ag.V.copy()))
                if d < tol or t > max_time: break
    else:
        while True:
            t0 = time.time()
            d = ag.sweep_gs() if variant.startswith('gs') else ag._sweep()
            t += time.time() - t0; n += 1
            if n % snap_every == 0: snaps.append((t, ag.V.copy()))
            if d < tol or t > max_time: break
    snaps.append((t, ag.V.copy()))
    return ag, snaps, n, t
