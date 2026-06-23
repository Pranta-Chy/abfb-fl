"""ABFB federated learning simulator (HAR / CIFAR-10 / DHCD).

Single configurable entry point for all six methods (FedAvg, PoC, Oort,
GT-LinUCB, ABFB, FedProx). Inherits the wireless / energy / LinFA Q-agent
machinery from fl_simulation_phase2 and adds the belief-filter and active-query
layer described in the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Sibling import for the baseline simulator
HERE = Path(__file__).resolve().parent
PHASE2_DIR = HERE.parent / "phase2_q1_extensions"
if str(PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE2_DIR))

import random                              # noqa: E402

import fl_simulation_phase2 as p2          # noqa: E402

# ABFB modules
from dhcd_loader import build_dhcd_loaders, DHCD_N_CLASSES   # noqa: E402
from dhcd_net import DHCDNet                                  # noqa: E402
from cifar10_loader import build_cifar10_loaders, CIFAR10_N_CLASSES  # noqa: E402
from cifar10_net import CIFAR10Net                                    # noqa: E402
from skeleton_belief_models import (                          # noqa: E402
    ServerBeliefStore, ActiveQueryTrigger,
    STATE_REPORT_BITS, SELECTION_ACK_BITS,
    QUERY_BUDGET, VOI_THRESHOLD_INIT,
)

# Monkey-patch the baseline get_model_class so DHCD support works inside
#    IoTClient.estimated_local_train_energy_j and elsewhere it's called. ────
_ORIG_GET_MODEL_CLASS = p2.get_model_class

def _phase3_get_model_class():
    """Dispatch including DHCD + CIFAR-10; falls back to the baseline dispatcher for mnist/har."""
    if p2.DATASET == "dhcd":
        return DHCDNet
    if p2.DATASET == "cifar10":
        return CIFAR10Net
    return _ORIG_GET_MODEL_CLASS()

p2.get_model_class = _phase3_get_model_class


# 1.  Configuration helpers
def apply_config(cfg: argparse.Namespace) -> None:
    """
    Mutate the baseline module globals to match this run's configuration.
    This is the simplest interop with the baseline global-driven architecture.
    """
    p2.DATASET = cfg.dataset
    p2.NUM_CLIENTS = cfg.n_clients
    p2.NUM_ROUNDS = cfg.rounds
    p2.K_SELECT = max(4, cfg.n_clients * 4 // 10)
    p2.Q_AGENT_TYPE = cfg.q_agent

    # Dataset-specific knobs preserved from the baseline; DHCD partition uses
    # the Dirichlet α (default 0.3) consistently.
    p2.HAR_DIRICHLET_A = cfg.dirichlet_alpha

    # Dataset-aware reward calibration  -  without this, ENERGY_NORM_J=10 (HAR-
    # tuned) is several orders of magnitude smaller than DHCD's per-round
    # energy (~5,000 J), making the LinFA reward effectively energy-blind on
    # DHCD. We scale the normalizer to the empirical per-round energy range
    # for each dataset so the energy/accuracy trade-off remains meaningful.
    _ENERGY_NORM_BY_DATASET = {
        "mnist":   10.0,    # MNIST CNN ~ 5 J/round
        "har":     10.0,    # HAR MLP ~ 5–20 J/round
        "cifar10": 1500.0,  # CIFAR10 CNN ~ 7,000–10,000 J/round (empirical, smoke test calibration)
        "dhcd":    1000.0,  # DHCD CNN ~ 5,000 J/round (scaled 100× from HAR)
    }
    p2.ENERGY_NORM_J = _ENERGY_NORM_BY_DATASET.get(cfg.dataset, 10.0)

    # Where this run's CSVs / figures land
    p2.OUT_DIR = (HERE / f"phase3_results_{cfg.dataset}_n{cfg.n_clients}"
                  / f"{cfg.method}_b{cfg.budget}_s{cfg.seed}")
    p2.OUT_DIR.mkdir(parents=True, exist_ok=True)
    p2.FIG_DIR = p2.OUT_DIR / "figures"
    p2.FIG_DIR.mkdir(exist_ok=True)


# 2.  Dataset / model dispatch (DHCD support)
def get_dataset_loaders_phase3(seed: int, cfg: argparse.Namespace):
    if cfg.dataset == "har":
        return p2.build_har_loaders(cfg.n_clients, p2.BATCH_SIZE, seed,
                                     dirichlet_alpha=cfg.dirichlet_alpha)
    if cfg.dataset == "dhcd":
        return build_dhcd_loaders(cfg.n_clients, p2.BATCH_SIZE, seed,
                                   dirichlet_alpha=cfg.dirichlet_alpha)
    if cfg.dataset == "cifar10":
        return build_cifar10_loaders(cfg.n_clients, p2.BATCH_SIZE, seed,
                                      dirichlet_alpha=cfg.dirichlet_alpha)
    if cfg.dataset == "mnist":
        return p2.build_mnist_loaders(
            cfg.n_clients, p2.CLASSES_PER_CLIENT, p2.BATCH_SIZE,
            p2.SAMPLES_PER_CLASS, seed)
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def get_model_class_phase3(cfg: argparse.Namespace):
    if cfg.dataset == "har":
        return p2.HARNet
    if cfg.dataset == "dhcd":
        return DHCDNet
    if cfg.dataset == "cifar10":
        return CIFAR10Net
    if cfg.dataset == "mnist":
        return p2.MNISTNet
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


# 3.  Helper: deterministic client-state initialization (shared across methods)
def make_initial_state(cfg):
    """Identical initial battery / RSSI / compute factor for every method
    within a (dataset, n_clients, seed) cell  -  guarantees fair comparison."""
    rng = random.Random(cfg.seed)
    init_batteries = [rng.uniform(60, 100) for _ in range(cfg.n_clients)]
    init_rssi      = [rng.uniform(-65, -45) for _ in range(cfg.n_clients)]
    init_compute   = [rng.uniform(0.5, 1.5) for _ in range(cfg.n_clients)]
    return init_batteries, init_rssi, init_compute


def build_clients(train_loaders, n_samples_per_client, cfg):
    """Build IoTClients with identical initial state across methods."""
    init_b, init_r, init_c = make_initial_state(cfg)
    return [p2.IoTClient(i, train_loaders[i], n_samples_per_client[i],
                          battery_pct=init_b[i], rssi=init_r[i],
                          compute_factor=init_c[i])
            for i in range(cfg.n_clients)]


# 4.  ABFB main loop  (mirrors p2.run_hybrid_rl with belief integration)
def run_abfb(clients, test_loader, tracker, cfg, verbose=True):
    """
    Active Belief-State Federated Bandit run.

    Differences from p2.run_hybrid_rl:
      (a) Server maintains ServerBeliefStore, updated from protocol observations.
      (b) Before LinUCB selection, ActiveQueryTrigger picks up to B clients
          to query for ground-truth state; query metadata energy is charged.
      (c) Selection uses belief context, not c.get_context().
      (d) After the round, belief updates from observed protocol signals
          (arrival, retx count, latency, observed train time, innovation).
    """
    if verbose:
        print("\n" + "=" * 70)
        print(f"  SIMULATION  -  ABFB (B={cfg.budget})  -  {cfg.dataset} / N={cfg.n_clients}")
        print("=" * 70)

    for c in clients:
        c.q_agent = p2.make_q_agent(c.id)

    model_cls = get_model_class_phase3(cfg)
    global_model = model_cls().to(p2.DEVICE)
    linucb = p2.LinUCBAgent(cfg.n_clients)

    belief_store = ServerBeliefStore(cfg.n_clients)
    query_trigger = ActiveQueryTrigger(
        voi_init=VOI_THRESHOLD_INIT,
        budget=cfg.budget,
    )

    accuracy_log, battery_log, per_class_log = [], [], []
    cm_final, reward_log = None, []
    query_rate_log, belief_rmse_log, metadata_bits_log = [], [], []
    prev_acc = 0.0

    for rnd in range(1, cfg.rounds + 1):
        is_warmup = rnd <= p2.WARMUP_ROUNDS
        t0 = time.time()

        for c in clients: c.step_environment()
        for c in clients:
            if not c.is_dead: c.drain_idle()

        alive_ids = [c.id for c in clients if not c.is_dead]

        # Initialize round counters BEFORE the active-query phase so that
        # query metadata energy (Tx state report + Rx selection ack) is
        # correctly attributed to the round-level totals  -  not silently
        local_deltas, weights, update_records = [], [], []
        round_bits_air = 0
        e_compute_round = e_tx_round = e_rx_round = 0.0
        e_aux_round = e_rl_round = 0.0
        n_skipped = n_retx_round = 0
        innov_acc = []

        # (a) Active query phase
        to_query = query_trigger.select_for_query(belief_store, alive_ids, linucb)
        round_meta_bits = 0
        for cid in to_query:
            c = clients[cid]
            # Server → client: selection/request ack  (client Rx)
            rx_meta_j = p2.EnergyModel.rx_energy_j(SELECTION_ACK_BITS)
            c.drain_j(rx_meta_j)
            e_rx_round += rx_meta_j
            # Client → server: state report           (client Tx)
            tx_meta_j = p2.EnergyModel.tx_energy_j(STATE_REPORT_BITS)
            c.drain_j(tx_meta_j)
            e_tx_round += tx_meta_j
            true_state = {
                "battery_normalized":  c.battery_pct / 100.0,
                "rssi_normalized":     (c.rssi + 90) / 60.0,
                "compute_normalized":  (c.compute_factor - 0.5) / 1.0,
            }
            belief_store.query_ground_truth(cid, true_state, rnd)
            round_meta_bits += SELECTION_ACK_BITS + STATE_REPORT_BITS

        # (b) Selection on belief context
        belief_contexts = {cid: belief_store.get_inferred_context(cid)
                            for cid in alive_ids}
        selected_ids = linucb.select_clients(belief_contexts, p2.K_SELECT)

        # 

        n_params = sum(p.numel() for p in global_model.parameters())
        downlink_bits = p2.count_param_bits(global_model, 32)

        for cid in selected_ids:
            c = clients[cid]
            if c.battery_pct < 5.0 and not is_warmup:
                c.staleness += 1; n_skipped += 1
                state = c.q_agent.get_state(c.battery_pct, c.rssi, 0.0, c.compute_factor)
                update_records.append({
                    "cid": cid, "state": state, "action": 0,
                    "bits_air": 0, "compute_j": 0.0, "tx_j": 0.0,
                    "rx_j": 0.0, "aux_j": 0.0, "tx_latency": 0.0,
                    "active_time_s": 0.0, "innov_norm": 0.0,
                    "transmitted": False, "n_retx": 0,
                })
                continue

            rx_j = c.drain_rx(downlink_bits); e_rx_round += rx_j
            local_model = c.local_train(global_model)
            train_compute_j = c.estimated_local_train_energy_j()
            e_compute_round += train_compute_j

            with torch.no_grad():
                vec_l = torch.cat([p.flatten() for p in local_model.parameters()])
                vec_g = torch.cat([p.flatten() for p in global_model.parameters()])
                innov_norm = float(torch.norm(vec_l - vec_g).cpu())
            innov_acc.append(innov_norm)
            innov_compute_j = c.drain_innov_norm_compute(n_params)
            e_aux_round += innov_compute_j

            state = c.q_agent.get_state(c.battery_pct, c.rssi, innov_norm, c.compute_factor)
            e_rl_round += c.drain_q_inference()
            action = c.q_agent.select_action(state, allow_skip=(not is_warmup))
            bits_per_param = p2.BIT_ACTIONS[action]

            if bits_per_param is None:
                c.staleness += 1; n_skipped += 1
                active_t = (c.active_time_for_rx(downlink_bits)
                            + c.active_time_for_compute(train_compute_j + innov_compute_j))
                c.refund_idle_for_active_period(active_t)
                update_records.append({
                    "cid": cid, "state": state, "action": action,
                    "bits_air": 0, "compute_j": train_compute_j, "tx_j": 0.0,
                    "rx_j": rx_j, "aux_j": innov_compute_j,
                    "tx_latency": 0.0, "active_time_s": active_t,
                    "innov_norm": innov_norm,
                    "transmitted": False, "n_retx": 0,
                })
                continue

            quant_compute_j = c.drain_quantize_compute(n_params)
            e_aux_round += quant_compute_j
            q_delta = p2.DeltaQuantizer.quantize_delta(local_model, global_model, bits_per_param)
            if p2.USE_DP:
                q_delta = p2.apply_dp_to_delta(q_delta)

            payload_bits = p2.count_param_bits(local_model, bits_per_param)
            success, bits_air, n_retx = p2.WirelessChannel.transmit(payload_bits, c.rssi)
            n_retx_round += n_retx
            tx_j = p2.EnergyModel.tx_energy_j(bits_air); e_tx_round += tx_j
            tx_latency = bits_air / p2.WIFI_BITRATE_BPS
            c.drain_j(tx_j); round_bits_air += bits_air

            active_t = (c.active_time_for_rx(downlink_bits)
                        + c.active_time_for_compute(train_compute_j + innov_compute_j + quant_compute_j)
                        + c.active_time_for_tx(bits_air))
            c.refund_idle_for_active_period(active_t)

            if success:
                w_innov = 1.0 + min(innov_norm / p2.INNOV_WEIGHT_NORM, 1.0)
                local_deltas.append(q_delta); weights.append(w_innov)
                c.staleness = 0
            else:
                c.staleness += 1

            update_records.append({
                "cid": cid, "state": state, "action": action,
                "bits_air": bits_air,
                "compute_j": train_compute_j, "tx_j": tx_j,
                "rx_j": rx_j, "aux_j": innov_compute_j + quant_compute_j,
                "tx_latency": tx_latency, "active_time_s": active_t,
                "innov_norm": innov_norm,
                "transmitted": success, "n_retx": n_retx,
            })

        if local_deltas:
            p2.aggregate_deltas(global_model, local_deltas, weights)

        acc, per_class, cm = p2.evaluate(global_model, test_loader)
        per_class_log.append(per_class); cm_final = cm
        acc_delta = acc - prev_acc; prev_acc = acc

        # Reward + LinUCB update
        round_rewards = []
        for r in update_records:
            cid = r["cid"]; c = clients[cid]
            reward = p2.compute_reward(r["bits_air"], acc_delta,
                                        r["compute_j"], r["tx_j"], r["tx_latency"])
            round_rewards.append(reward)
            next_state = c.q_agent.get_state(c.battery_pct, c.rssi, r["innov_norm"], c.compute_factor)
            c.q_agent.update(r["state"], r["action"], reward, next_state)
            e_rl_round += c.drain_q_update()
            c.q_agent.decay()
            linucb_r = acc_delta if r["transmitted"] else 0.0
            linucb.update(cid, belief_contexts.get(cid, np.zeros(p2.CONTEXT_DIM)), linucb_r)

        # (d) Post-round belief update from observations
        observed_by_cid = {r["cid"]: r for r in update_records}
        for cid in alive_ids:
            if cid in observed_by_cid and observed_by_cid[cid]["transmitted"]:
                r = observed_by_cid[cid]
                # Expected train time = active_time_s sans Tx component
                expected_t = max(1e-3, r["active_time_s"] - (r["bits_air"] / p2.WIFI_BITRATE_BPS))
                obs = {
                    "update_arrived":          True,
                    "bits_received":           r["bits_air"],
                    "latency_s":               r["tx_latency"],
                    "n_retx":                  r["n_retx"],
                    "innov_norm":              r["innov_norm"],
                    "observed_train_time_s":   r["active_time_s"],
                    "expected_train_time_s":   expected_t,
                }
            else:
                obs = {"update_arrived": False}
            belief_store.update_from_observation(cid, obs)

        # Update staleness for unselected
        for cid in range(cfg.n_clients):
            if cid not in selected_ids and not clients[cid].is_dead:
                clients[cid].staleness += 1

        # Belief-tracking error
        bel_err = []
        for cid in alive_ids:
            c = clients[cid]
            true_bat = c.battery_pct / 100.0
            true_ch  = (c.rssi + 90) / 60.0
            true_cp  = (c.compute_factor - 0.5) / 1.0
            b = belief_store.belief[cid]
            bel_err.append([(b.battery.mean - true_bat),
                            (b.channel.mean - true_ch),
                            (b.compute.mean - true_cp)])
        bel_err_arr = np.array(bel_err) if bel_err else np.zeros((1, 3))
        rmse_per_component = np.sqrt(np.mean(bel_err_arr ** 2, axis=0)).tolist()

        # Round logging
        avg_battery = float(np.mean([c.battery_pct for c in clients]))
        n_dead = sum(1 for c in clients if c.is_dead)
        total_active_s = sum(r.get("active_time_s", 0.0) for r in update_records)
        n_alive = sum(1 for c in clients if not c.is_dead)
        e_idle_gross = p2.GATEWAY_IDLE_POWER_W * p2.ROUND_WALL_CLOCK_S * n_alive
        # Clamp idle ≥ 0: if total_active_s > ROUND_WALL_CLOCK_S × n_alive due to
        # rounding / accumulation, the gross idle can underflow.  (Mirrors the baseline.)
        e_idle_round = max(0.0,
                           e_idle_gross - p2.GATEWAY_IDLE_POWER_W * total_active_s)
        energy_round = (e_compute_round + e_tx_round + e_rx_round
                         + e_idle_round + e_aux_round + e_rl_round)

        tracker.log_round(
            rnd, acc, avg_battery, round_bits_air + round_meta_bits,
            energy_round, e_compute_round, e_tx_round, e_rx_round,
            e_idle_round, e_aux_round, e_rl_round,
            selected_ids, n_skipped, n_dead, n_retx_round,
            float(np.mean(innov_acc)) if innov_acc else 0.0,
            float(np.mean(round_rewards)) if round_rewards else 0.0,
        )

        for cid, c in enumerate(clients):
            rec = next((r for r in update_records if r["cid"] == cid), None)
            tracker.log_client_state(rnd, cid, c.battery_pct, c.rssi,
                                     cid in selected_ids,
                                     rec["action"] if rec else None,
                                     rec["bits_air"] if rec else 0,
                                     (rec["compute_j"] + rec["tx_j"]) if rec else 0.0)

        accuracy_log.append(acc * 100); battery_log.append(avg_battery)
        reward_log.append(float(np.mean(round_rewards)) if round_rewards else 0.0)
        query_rate_log.append(len(to_query) / max(1, len(alive_ids)))
        belief_rmse_log.append(rmse_per_component)
        metadata_bits_log.append(round_meta_bits)

        if verbose and (rnd <= 10 or rnd % 10 == 0 or rnd == cfg.rounds):
            tag = " [WARMUP]" if is_warmup else ""
            print(f"  R{rnd:3d}{tag:9s} | Acc {acc*100:6.2f}% | "
                  f"E {energy_round:6.2f}J | Bat {avg_battery:5.1f}% | "
                  f"Q {len(to_query)} | Skip {n_skipped} | "
                  f"Dead {n_dead} | {time.time()-t0:.1f}s")

    return {
        "accuracy_log": accuracy_log,
        "battery_log": battery_log,
        "per_class_log": per_class_log,
        "cm_final": cm_final,
        "reward_log": reward_log,
        "query_rate_log": query_rate_log,
        "belief_rmse_log": belief_rmse_log,
        "metadata_bits_log": metadata_bits_log,
        "n_queries_total": belief_store.n_queries_total,
        "linucb_counts": linucb.selection_counts.copy(),
        "linucb_jain": linucb.jain_fairness(),
        "q_policies": [c.q_agent.policy_table().copy() for c in clients],
        "action_dists": [c.q_agent.get_action_distribution() for c in clients],
    }


# 5.  Power-of-Choice baseline (Cho et al. 2020)
def run_poc(clients, test_loader, tracker, cfg, verbose=True, poc_d=None):
    """
    Power-of-Choice: sample d ≥ K candidates uniformly, then pick the K with
    highest local loss. d defaults to 2K.
    """
    if poc_d is None:
        poc_d = min(2 * p2.K_SELECT, cfg.n_clients)

    if verbose:
        print("\n" + "=" * 70)
        print(f"  SIMULATION  -  Power-of-Choice (d={poc_d})  -  {cfg.dataset} / N={cfg.n_clients}")
        print("=" * 70)

    model_cls = get_model_class_phase3(cfg)
    global_model = model_cls().to(p2.DEVICE)
    accuracy_log, battery_log = [], []
    per_class_log = []; cm_final = None
    prev_acc = 0.0

    for rnd in range(1, cfg.rounds + 1):
        t0 = time.time()
        for c in clients: c.step_environment()
        for c in clients:
            if not c.is_dead: c.drain_idle()
        alive = [c.id for c in clients if not c.is_dead]
        if not alive:
            break

        # Initialize round counters BEFORE the probe loop so probe Rx + probe
        # compute are correctly attributed to the round-level totals.
        # (Bug fix: probe energy was previously drained from battery but not
        # logged, undercounting PoC's true overhead.)
        local_models, weights, update_records = [], [], []
        round_bits_air = 0
        e_compute_round = e_tx_round = e_rx_round = 0.0
        n_retx_round = 0

        # 1) Sample d candidates uniformly
        rng = np.random.RandomState(cfg.seed + rnd)
        cand = list(rng.choice(alive, size=min(poc_d, len(alive)), replace=False))

        # 2) Score each candidate by current local-batch loss (server triggers
        #    a small local probe; cost charged via downlink + compute estimate)
        scores = {}
        n_params = sum(p.numel() for p in global_model.parameters())
        downlink_bits = p2.count_param_bits(global_model, 32)
        for cid in cand:
            c = clients[cid]
            rx_j = c.drain_rx(downlink_bits)
            e_rx_round += rx_j
            try:
                # Single-batch forward → loss
                global_model.eval()
                with torch.no_grad():
                    xb, yb = next(iter(c.data_loader))
                    xb = xb.to(p2.DEVICE); yb = yb.to(p2.DEVICE)
                    out = global_model(xb)
                    loss = torch.nn.functional.cross_entropy(out, yb).item()
                scores[cid] = loss
            except (StopIteration, Exception):
                scores[cid] = 0.0
            # Approx probe compute energy (1 forward batch)
            probe_macs = p2.estimate_compute_macs(
                n_samples=p2.BATCH_SIZE, epochs=1,
                batch_size=p2.BATCH_SIZE, model_cls=model_cls) // p2.LOCAL_EPOCHS
            probe_j = (probe_macs * p2.SECONDS_PER_MAC * p2.GATEWAY_CPU_POWER_W
                       / p2.PSU_EFFICIENCY)
            c.drain_j(probe_j)
            e_compute_round += probe_j

        # Restore train mode on the global model (the probe loop above set eval();
        # IoTClient.local_train deepcopies global_model, so a leaked eval state
        # could otherwise propagate into local training via BN/Dropout layers).
        global_model.train()

        # 3) Pick top-K by loss
        selected_ids = sorted(scores.keys(), key=lambda i: scores[i],
                              reverse=True)[:p2.K_SELECT]

        for cid in selected_ids:
            c = clients[cid]
            rx_j = c.drain_rx(downlink_bits); e_rx_round += rx_j
            local_model = c.local_train(global_model)
            train_j = c.estimated_local_train_energy_j(); e_compute_round += train_j
            payload_bits = p2.count_param_bits(local_model, 32)
            success, bits_air, n_retx = p2.WirelessChannel.transmit(payload_bits, c.rssi)
            n_retx_round += n_retx
            tx_j = p2.EnergyModel.tx_energy_j(bits_air); e_tx_round += tx_j
            c.drain_j(tx_j); round_bits_air += bits_air
            active_t = (c.active_time_for_rx(downlink_bits)
                        + c.active_time_for_compute(train_j)
                        + c.active_time_for_tx(bits_air))
            c.refund_idle_for_active_period(active_t)
            if success:
                local_models.append(local_model)
                weights.append(c.n_samples)
            update_records.append({
                "cid": cid, "bits_air": bits_air, "compute_j": train_j,
                "tx_j": tx_j, "rx_j": rx_j, "tx_latency": bits_air / p2.WIFI_BITRATE_BPS,
                "active_time_s": active_t, "transmitted": success,
            })

        if local_models:
            p2.fedavg_full_aggregate(global_model, local_models, weights)

        acc, per_class, cm = p2.evaluate(global_model, test_loader)
        per_class_log.append(per_class); cm_final = cm
        prev_acc = acc

        avg_battery = float(np.mean([c.battery_pct for c in clients]))
        n_dead = sum(1 for c in clients if c.is_dead)
        n_alive = sum(1 for c in clients if not c.is_dead)
        total_active_s = sum(r.get("active_time_s", 0.0) for r in update_records)
        # Same defensive clamp as ABFB above (baseline parity).
        e_idle_round = max(
            0.0,
            p2.GATEWAY_IDLE_POWER_W * (p2.ROUND_WALL_CLOCK_S * n_alive - total_active_s),
        )
        energy_round = e_compute_round + e_tx_round + e_rx_round + e_idle_round

        tracker.log_round(
            rnd, acc, avg_battery, round_bits_air,
            energy_round, e_compute_round, e_tx_round, e_rx_round,
            e_idle_round, 0.0, 0.0,
            selected_ids, 0, n_dead, n_retx_round, 0.0, 0.0,
        )
        for cid, c in enumerate(clients):
            rec = next((r for r in update_records if r["cid"] == cid), None)
            tracker.log_client_state(rnd, cid, c.battery_pct, c.rssi,
                                     cid in selected_ids, None,
                                     rec["bits_air"] if rec else 0,
                                     (rec["compute_j"] + rec["tx_j"]) if rec else 0.0)
        # Staleness update for unselected alive clients  -  keeps environment
        # state consistent with run_abfb / run_hybrid_rl even though PoC
        # itself does not consume staleness.
        for cid in range(cfg.n_clients):
            if cid not in selected_ids and not clients[cid].is_dead:
                clients[cid].staleness += 1

        accuracy_log.append(acc * 100); battery_log.append(avg_battery)

        if verbose and (rnd <= 10 or rnd % 10 == 0 or rnd == cfg.rounds):
            print(f"  R{rnd:3d}      | Acc {acc*100:6.2f}% | E {energy_round:6.2f}J | "
                  f"Bat {avg_battery:5.1f}% | Dead {n_dead} | {time.time()-t0:.1f}s")

    return {
        "accuracy_log": accuracy_log, "battery_log": battery_log,
        "per_class_log": per_class_log, "cm_final": cm_final,
    }


# 6.  Dispatcher
METHOD_DISPATCH = {
    "fedavg":    lambda clients, tl, tr, cfg, v:
                    p2.run_fedavg_baseline(clients, tl, tr, verbose=v,
                                            fedprox_mu=0.0, name="FedAvg"),
    "fedprox":   lambda clients, tl, tr, cfg, v:
                    p2.run_fedavg_baseline(clients, tl, tr, verbose=v,
                                            fedprox_mu=p2.FEDPROX_MU, name="FedProx"),
    "oort":      lambda clients, tl, tr, cfg, v:
                    p2.run_oort_baseline(clients, tl, tr, verbose=v),
    "poc":       lambda clients, tl, tr, cfg, v:
                    run_poc(clients, tl, tr, cfg, verbose=v),
    "gt_linucb": lambda clients, tl, tr, cfg, v:
                    p2.run_hybrid_rl(clients, tl, tr, verbose=v),
    "abfb":      lambda clients, tl, tr, cfg, v:
                    run_abfb(clients, tl, tr, cfg, verbose=v),
}


# 7.  Main
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ABFB FL simulator")
    ap.add_argument("--method", required=True,
                    choices=list(METHOD_DISPATCH.keys()))
    ap.add_argument("--dataset", required=True, choices=["har", "dhcd", "cifar10", "mnist"])
    ap.add_argument("--n_clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--budget", type=int, default=QUERY_BUDGET,
                    help="ABFB per-round query budget (ignored for non-ABFB)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dirichlet_alpha", type=float, default=0.3)
    ap.add_argument("--q_agent", default="linfa", choices=["linfa", "tabular", "dqn"])
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    return ap.parse_args()


_METHOD_TO_SYSTEM_NAME = {
    "fedavg":    "FedAvg",
    "fedprox":   "FedProx",
    "poc":       "PowerOfChoice",
    "oort":      "Oort",
    "gt_linucb": "Hybrid-RL",
    "abfb":      "ABFB",
}


def main():
    cfg = parse_args()
    apply_config(cfg)

    p2.set_global_seed(cfg.seed)
    print(f"\n[Phase3] method={cfg.method} dataset={cfg.dataset} "
          f"N={cfg.n_clients} rounds={cfg.rounds} budget={cfg.budget} seed={cfg.seed}")
    print(f"[Phase3] OUT_DIR={p2.OUT_DIR}")

    # Loaders + clients (deterministic state init shared across methods)
    train_loaders, test_loader, ns = get_dataset_loaders_phase3(cfg.seed, cfg)
    clients = build_clients(train_loaders, ns, cfg)
    sys_name = _METHOD_TO_SYSTEM_NAME[cfg.method]
    tracker = p2.MetricsTracker(sys_name)

    # Save run config for reproducibility
    cfg_dict = vars(cfg).copy()
    cfg_dict["out_dir"] = str(p2.OUT_DIR)
    with open(p2.OUT_DIR / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Run
    t_start = time.time()
    result = METHOD_DISPATCH[cfg.method](clients, test_loader, tracker, cfg, cfg.verbose)
    t_elapsed = time.time() - t_start

    # Persist round-level CSVs via The tracker
    tracker.save_all(p2.OUT_DIR)

    # Persist ABFB-specific logs
    if cfg.method == "abfb":
        import pandas as pd
        belief_rmse_df = pd.DataFrame(
            result["belief_rmse_log"],
            columns=["rmse_battery", "rmse_channel", "rmse_compute"],
        )
        belief_rmse_df.insert(0, "round", range(1, len(belief_rmse_df) + 1))
        belief_rmse_df.to_csv(p2.OUT_DIR / "belief_log.csv", index=False)

        query_df = pd.DataFrame({
            "round": range(1, len(result["query_rate_log"]) + 1),
            "query_rate": result["query_rate_log"],
            "metadata_bits": result["metadata_bits_log"],
        })
        query_df.to_csv(p2.OUT_DIR / "query_log.csv", index=False)

    final_acc = result["accuracy_log"][-1] if result["accuracy_log"] else 0.0
    summary = {
        "method": cfg.method,
        "dataset": cfg.dataset,
        "n_clients": cfg.n_clients,
        "budget": cfg.budget,
        "seed": cfg.seed,
        "rounds": cfg.rounds,
        "final_accuracy": final_acc,
        "wall_time_s": t_elapsed,
        "cumulative_bits": int(tracker._bits_cum),
        "cumulative_energy_j": float(tracker._energy_cum),
    }
    if cfg.method == "abfb":
        summary["total_queries"] = result["n_queries_total"]
        summary["mean_query_rate"] = float(np.mean(result["query_rate_log"]))
        summary["final_belief_rmse"] = result["belief_rmse_log"][-1]
        summary["total_metadata_bits"] = int(np.sum(result["metadata_bits_log"]))

    with open(p2.OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Phase3] Done in {t_elapsed/60:.1f} min  -  final acc {final_acc:.2f}%")
    print(f"[Phase3] Summary written to {p2.OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()