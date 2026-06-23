"""Belief models and server-side belief store for ABFB.

Contains the three per-client Bayesian belief filters (battery, channel, compute),
the per-client belief container, the ServerBeliefStore that holds them all, and
the ActiveQueryTrigger that decides which clients to query each round.
"""
import numpy as np
import random
from collections import defaultdict


# Energy / channel constants
GATEWAY_IDLE_POWER_W = 0.264      # W
ROUND_WALL_CLOCK_S   = 30.0       # s
BATTERY_CAPACITY_J   = 33_300.0   # J
LINUCB_ALPHA         = 1.0

# ABFB hyperparameters
VOI_THRESHOLD_INIT = 0.05    # τ initial value before percentile auto-tune kicks in
VOI_PERCENTILE     = 75      # adaptive τ = p-th percentile of historical VoI
VOI_HISTORY_LEN    = 200     # rolling window for τ calibration
QUERY_BUDGET       = 2       # B  -  max queries per round across all clients
COLD_START_OBS     = 5       # rounds of observations before belief is "warm"

# Belief decay/observation parameters
BATTERY_PASSIVE_DRAIN = (GATEWAY_IDLE_POWER_W * ROUND_WALL_CLOCK_S
                          / BATTERY_CAPACITY_J)
BATTERY_ACTIVE_DRAIN  = 8.0 / BATTERY_CAPACITY_J   # ~8 J avg compute energy
BATTERY_DRIFT_VAR     = 1e-5         # process noise per round (σ²_drift)
BATTERY_MIN_THRESHOLD = 0.10         # client cannot participate below this
TRUNC_GAUSSIAN_LAMBDA = 0.40         # moment-matched correction for censored obs

CHANNEL_EMA_ALPHA = 0.3      # exponential moving average weight
CHANNEL_W_RETX    = 0.7      # weight on retx-based quality proxy
CHANNEL_W_LAT     = 0.3      # weight on latency-based quality proxy
CHANNEL_MAX_LAT_S = 5.0      # latency at which channel proxy → 0
COMPUTE_EMA_ALPHA = 0.3

# Per-belief prior variances (used to normalize total_variance to unitless scale)
PRIOR_VAR_BATTERY = 0.04
PRIOR_VAR_CHANNEL = 0.10
PRIOR_VAR_COMPUTE = 0.10

# Metadata costs (in bits)
STATE_REPORT_BITS    = 192   # 4 × fp32 + ~64 bit framing
SELECTION_ACK_BITS   = 80


# 1.  BATTERY BELIEF  (Bayesian filter, drift + observation model)
class BatteryBelief:
    """
    Belief over normalized battery level in [0, 1]  -  Bayesian filter.

    Two-step update:
        Predict (drift):       μ⁻ = μ - δ_drain,  σ²⁻ = σ² + σ²_drift
        Observation update:    participation is a censored measurement
                                𝟙{s ≥ s_min}. Moment-matched truncated-Gaussian
                                correction:
                                    μ = max(μ⁻, s_min + λ σ⁻),
                                    σ² = ρ · σ²⁻,  ρ < 1
        Decay (no obs):        no observation update; variance keeps growing.
    """
    def __init__(self, prior_mean=0.8, prior_var=PRIOR_VAR_BATTERY):
        self.mean = prior_mean
        self.variance = prior_var
        self.n_observations = 0

    def _predict(self, active):
        """Drift step: drain + process noise."""
        drain = BATTERY_ACTIVE_DRAIN if active else BATTERY_PASSIVE_DRAIN
        self.mean = max(0.0, self.mean - drain)
        self.variance = min(0.25, self.variance + BATTERY_DRIFT_VAR)

    def observe_participation(self):
        """
        Predict + censored observation update.
        Participation implies battery ≥ BATTERY_MIN_THRESHOLD.
        """
        self._predict(active=True)
        sigma = np.sqrt(self.variance)
        # Truncated-Gaussian moment matching: lift mean if it was too low.
        lifted = BATTERY_MIN_THRESHOLD + TRUNC_GAUSSIAN_LAMBDA * sigma
        self.mean = float(np.clip(max(self.mean, lifted), 0.0, 1.0))
        self.variance = max(0.005, self.variance * 0.90)   # information gain
        self.n_observations += 1

    def decay(self):
        """No observation this round  -  only the predict step runs."""
        self._predict(active=False)

    def snap_to(self, true_value):
        """Active query result: reset to ground truth with low variance."""
        self.mean = float(np.clip(true_value, 0.0, 1.0))
        self.variance = 0.005


# 2.  CHANNEL BELIEF  (EMA over inferred channel quality)
class ChannelBelief:
    """
    Belief over normalized channel quality (proxy for RSSI in [0, 1]).
    Two-signal EMA: weighted combination of retx-based and latency-based proxies.

        q_retx = 1 / (1 + n_retx)
        q_lat  = 1 - min(latency_s / CHANNEL_MAX_LAT_S, 1)
        q_hat  = w_retx · q_retx + w_lat · q_lat
        μ      = α · q_hat + (1 - α) · μ
    """
    def __init__(self, prior_mean=0.5, prior_var=PRIOR_VAR_CHANNEL):
        self.mean = prior_mean
        self.variance = prior_var

    def observe(self, n_retx, latency_s):
        q_retx = 1.0 / (1.0 + max(0, n_retx))
        q_lat  = 1.0 - min(max(latency_s, 0.0) / CHANNEL_MAX_LAT_S, 1.0)
        q_hat  = CHANNEL_W_RETX * q_retx + CHANNEL_W_LAT * q_lat
        self.mean = CHANNEL_EMA_ALPHA * q_hat + (1 - CHANNEL_EMA_ALPHA) * self.mean
        self.variance = max(0.005, self.variance * 0.95)

    def decay(self):
        self.variance = min(0.25, self.variance * 1.05)

    def snap_to(self, true_value):
        self.mean = float(np.clip(true_value, 0.0, 1.0))
        self.variance = 0.005


# 3.  COMPUTE BELIEF  (EMA over inferred compute capacity)
class ComputeBelief:
    """
    Belief over normalized compute_factor in [0, 1].
    Inferred from observed training time vs expected.
    """
    def __init__(self, prior_mean=0.5, prior_var=PRIOR_VAR_COMPUTE):
        self.mean = prior_mean
        self.variance = prior_var

    def observe(self, observed_train_time_s, expected_train_time_s):
        """observed = expected × compute_factor → factor = observed / expected."""
        if expected_train_time_s > 0:
            factor = observed_train_time_s / expected_train_time_s
            factor = max(0.5, min(1.5, factor))
            normalized = (factor - 0.5) / 1.0     # → [0, 1]
            self.mean = COMPUTE_EMA_ALPHA * normalized + (1 - COMPUTE_EMA_ALPHA) * self.mean
            self.variance = max(0.005, self.variance * 0.95)

    def decay(self):
        self.variance = min(0.25, self.variance * 1.05)

    def snap_to(self, true_value):
        self.mean = float(np.clip(true_value, 0.0, 1.0))
        self.variance = 0.005


# 4.  PER-CLIENT BELIEF  (container)
class ClientBelief:
    """Server's belief over a single client's hidden state."""
    def __init__(self, cid):
        self.cid = cid
        self.battery = BatteryBelief()
        self.channel = ChannelBelief()
        self.compute = ComputeBelief()
        self.staleness = 0
        self.n_observations = 0
        self.is_new = True            # cold-start flag

    def update_from_obs(self, obs):
        """obs dict: {update_arrived, n_retx, latency_s, expected_train_time_s, innov_norm}"""
        if obs.get("update_arrived", False):
            self.battery.observe_participation()
            self.channel.observe(obs.get("n_retx", 0),
                                  obs.get("latency_s", 0.0))
            if obs.get("expected_train_time_s", 0) > 0:
                self.compute.observe(obs.get("observed_train_time_s", 0),
                                      obs["expected_train_time_s"])
            self.staleness = 0
            self.n_observations += 1
            if self.n_observations >= COLD_START_OBS:
                self.is_new = False
        else:
            self.battery.decay()
            self.channel.decay()
            self.compute.decay()
            self.staleness += 1

    def snap_to_ground_truth(self, true_state):
        """Active query: reset all beliefs to ground truth."""
        self.battery.snap_to(true_state["battery_normalized"])
        self.channel.snap_to(true_state["rssi_normalized"])
        self.compute.snap_to(true_state["compute_normalized"])
        self.staleness = 0
        self.is_new = False

    def total_variance(self):
        """
        Unitless aggregate variance: each component normalized by its prior
        variance so battery / channel / compute contribute on a comparable scale.
        Range ≈ [0, 3] (≈ 1 at initialization for each well-formed prior).
        """
        return (self.battery.variance / PRIOR_VAR_BATTERY
                + self.channel.variance / PRIOR_VAR_CHANNEL
                + self.compute.variance / PRIOR_VAR_COMPUTE)

    def to_context(self):
        """4-D context vector matching baseline signature."""
        return np.array([
            self.battery.mean,
            self.channel.mean,
            min(self.staleness, 10) / 10.0,
            self.compute.mean,
        ], dtype=np.float64)


# 5.  SERVER BELIEF STORE  (central object held by the server)
class ServerBeliefStore:
    """
    Server-side belief storage for all N clients.

    Belief is updated by:
        (a) Passive observation  -  every round, from FL protocol signals.
        (b) Active query  -  on demand, when VoI(c) > τ.

    Note: this class is intentionally simple. Bayesian filters could be
    upgraded to particle filters or Kalman filters if belief tracking error
    is too high.
    """
    def __init__(self, n_clients):
        self.n_clients = n_clients
        self.belief = {cid: ClientBelief(cid) for cid in range(n_clients)}
        self.observation_log = []
        self.query_log = []                 # list of (round, cid, voi)
        self.n_queries_total = 0

    def update_from_observation(self, cid, obs):
        """Passive update from FL protocol signals."""
        self.belief[cid].update_from_obs(obs)
        self.observation_log.append((cid, obs))

    def query_ground_truth(self, cid, true_state, rnd):
        """
        Active query: server requests state report, gets ground truth.
        Returns the bits transmitted for this query (for energy accounting).
        """
        self.belief[cid].snap_to_ground_truth(true_state)
        self.query_log.append((rnd, cid, self.belief[cid].total_variance()))
        self.n_queries_total += 1
        # Caller charges the metadata Tx/Rx energy
        return STATE_REPORT_BITS, SELECTION_ACK_BITS

    def get_inferred_context(self, cid):
        return self.belief[cid].to_context()

    def get_belief_variance(self, cid):
        return self.belief[cid].total_variance()

    def get_all_contexts(self, alive_ids):
        return {cid: self.get_inferred_context(cid) for cid in alive_ids}

    def belief_summary(self):
        """For logging / analysis."""
        return {
            "total_queries": self.n_queries_total,
            "avg_total_variance": np.mean([b.total_variance()
                                            for b in self.belief.values()]),
            "n_new_clients": sum(1 for b in self.belief.values() if b.is_new),
        }


# 6.  ACTIVE QUERY TRIGGER  (Value-of-Information driven)
class ActiveQueryTrigger:
    """
    Decides which clients to query for ground truth this round.

    VoI(c) = trace(Σ_c) × |UCB_bonus(c)|     [Taylor-expansion surrogate]

    Reasoning:
        - High belief variance → server is uncertain about this client.
        - High UCB bonus      → this client is a candidate for selection.
        - Their product       → expected utility gain from querying.

    Adaptive threshold:
        τ is the running p-th percentile (default 75) of historical VoI scores.
        This auto-tunes to the current belief distribution and avoids manual
        calibration. Falls back to VOI_THRESHOLD_INIT before history is warm.

    The query budget B caps the number of queries per round.
    Cold-start clients are queried first (within budget) regardless of VoI.
    """
    def __init__(self, voi_init=VOI_THRESHOLD_INIT, budget=QUERY_BUDGET,
                 percentile=VOI_PERCENTILE, history_len=VOI_HISTORY_LEN):
        self.tau = voi_init
        self.B = budget
        self.percentile = percentile
        self.history_len = history_len
        self.voi_history = []          # rolling window of VoI samples

    def _update_threshold(self):
        if len(self.voi_history) >= 20:
            self.tau = float(np.percentile(self.voi_history, self.percentile))

    def select_for_query(self, belief_store, candidate_ids, linucb_agent):
        """
        Returns the list of client IDs to actively query this round.
        Cold-start clients are prioritized within the budget B.
        """
        # Step 1: prioritize new clients
        new_clients = [cid for cid in candidate_ids
                        if belief_store.belief[cid].is_new]

        # Step 2: compute VoI for warm clients
        vois = {}
        for cid in candidate_ids:
            if cid in new_clients:
                continue
            x = belief_store.get_inferred_context(cid)
            var = belief_store.get_belief_variance(cid)
            try:
                A_inv = np.linalg.inv(linucb_agent.A[cid])
                ucb = LINUCB_ALPHA * float(np.sqrt(x @ A_inv @ x))
            except Exception:
                ucb = LINUCB_ALPHA
            voi_score = var * abs(ucb)
            vois[cid] = voi_score
            # Record for adaptive threshold calibration
            self.voi_history.append(voi_score)
            if len(self.voi_history) > self.history_len:
                self.voi_history.pop(0)

        # Refresh τ from recent history before applying it
        self._update_threshold()

        sorted_warm = sorted(vois.items(), key=lambda x: x[1], reverse=True)

        # Step 3: pick new clients up to budget, then top-VoI warm clients
        to_query = list(new_clients[:self.B])
        remaining_budget = self.B - len(to_query)
        for cid, voi in sorted_warm[:remaining_budget]:
            if voi > self.tau:
                to_query.append(cid)
        return to_query


# 7.  MAIN INTEGRATION POINT  (called from the simulation loop)
def phase3_round(clients, belief_store, query_trigger, linucb,
                  K_select, rnd):
    """
    Single round of the ABFB Active Belief Hybrid-RL loop.

    Returns:
        selected_ids       : list of client IDs selected this round
        queries_issued     : list of client IDs queried this round
        metadata_bits      : total metadata bits transmitted (for energy)

    This function replaces the LinUCB selection step in the baseline run_hybrid_rl.
    The rest of the round (training, Tx, aggregation, evaluation) is unchanged.
    """
    alive_ids = [c.id for c in clients if not c.is_dead]

    # Active query phase
    to_query = query_trigger.select_for_query(belief_store, alive_ids, linucb)
    queries_issued = []
    metadata_bits = 0
    for cid in to_query:
        c = clients[cid]
        # The server "knows" the ground truth via the simulator (this is the
        # simulator's privilege; in real deployment, this would be a uplink Rx).
        true_state = {
            "battery_normalized": c.battery_pct / 100.0,
            "rssi_normalized":    (c.rssi + 90) / 60.0,
            "compute_normalized": (c.compute_factor - 0.5) / 1.0,
        }
        tx_bits, rx_bits = belief_store.query_ground_truth(cid, true_state, rnd)
        # Charge metadata energy (client uploads state, server replies with ack)
        # Caller: charge c.drain_j(EnergyModel.tx_energy_j(tx_bits))
        # Caller: charge c.drain_j(EnergyModel.rx_energy_j(rx_bits))
        metadata_bits += tx_bits + rx_bits
        queries_issued.append(cid)

    # Selection phase
    contexts = belief_store.get_all_contexts(alive_ids)
    selected_ids = linucb.select_clients(contexts, K_select)

    return selected_ids, queries_issued, metadata_bits


# 8.  POST-ROUND BELIEF UPDATE  (called after Tx phase)
def phase3_update_beliefs(belief_store, alive_ids, selected_ids,
                            update_records):
    """
    After the round's Tx phase completes, update belief from observations.

    update_records (from the baseline run_hybrid_rl) contains per-client outcome
    info: which clients transmitted, with what payload, latency, and retx count.

    For non-selected clients, the belief decays (passive drain + variance growth).
    """
    transmitted_cids = {r["cid"] for r in update_records if r.get("transmitted")}

    for cid in alive_ids:
        if cid in transmitted_cids:
            rec = next(r for r in update_records if r["cid"] == cid)
            obs = {
                "update_arrived": True,
                "bits_received":  rec.get("bits_air", 0),
                "n_retx":         rec.get("n_retx", 0),
                "latency_s":      rec.get("tx_latency", 0.0),
                "innov_norm":     rec.get("innov_norm", 0.0),
                "observed_train_time_s": rec.get("active_time_s", 0.0),
                "expected_train_time_s": rec.get("active_time_s", 0.0),
            }
            belief_store.update_from_observation(cid, obs)
        else:
            belief_store.update_from_observation(cid,
                                                   {"update_arrived": False})


# Quick smoke test
if __name__ == "__main__":
    print("ABFB belief models  -  smoke test")
    store = ServerBeliefStore(n_clients=10)
    print(f"  Initial belief mean (client 0): {store.get_inferred_context(0)}")
    print(f"  Initial total variance: {store.get_belief_variance(0):.4f}")

    # Simulate 5 successful observations
    for _ in range(5):
        store.update_from_observation(0, {
            "update_arrived": True,
            "n_retx": 2,
            "latency_s": 0.5,
            "observed_train_time_s": 7.0,
            "expected_train_time_s": 7.0,
        })
    print(f"  After 5 obs (client 0): {store.get_inferred_context(0)}")
    print(f"  Variance: {store.get_belief_variance(0):.4f}")
    print(f"  is_new flag: {store.belief[0].is_new}")

    # Simulate 5 misses
    for _ in range(5):
        store.update_from_observation(0, {"update_arrived": False})
    print(f"  After 5 misses (client 0): {store.get_inferred_context(0)}")

    print("\nSmoke test OK.")
