"""
simulation.py
==============
Car-following simulation: our car (ego) follows a lead car that suddenly
brakes hard. The braking assist reads (speed, gap, drowsiness) every 50 ms
and applies a braking force. We compare FUZZY vs CRISP controllers.

Physics (simple point-mass model)
  max deceleration at 100 % brake  : 9.0 m/s^2  (~0.9 g, dry road)
  brake < 5 % (dead zone) -> smooth cruise control back to set speed (<=1.5 m/s^2)
  brake actuator lag               : first-order, time constant 0.15 s
"""

import numpy as np
from braking_system import fuzzy_brake, crisp_brake

DT = 0.05          # s
T_END = 20.0       # s
A_MAX = 9.0        # m/s^2 at 100 % brake
A_CRUISE = 1.5     # m/s^2 re-acceleration
TAU = 0.15         # s brake actuator lag
DEAD_ZONE = 5.0    # % - brake commands below this are ignored (pedal free play)
K_CRUISE = 0.5     # 1/s cruise-control gain
KMH = 1 / 3.6


def lead_profile(t, cruise_kmh=80, brake_at=4.0, decel=7.0, final_kmh=10):
    """Lead car cruises, then brakes hard down to a crawl (sudden traffic jam)."""
    v0, vf = cruise_kmh * KMH, final_kmh * KMH
    if t < brake_at:
        return v0
    return max(vf, v0 - decel * (t - brake_at))


def simulate(controller, drowsiness=2.0, set_kmh=80, gap0=45.0, **lead_kw):
    n = int(T_END / DT)
    t = np.arange(n) * DT
    v = np.zeros(n); gap = np.zeros(n); b_cmd = np.zeros(n); b_act = np.zeros(n)
    acc = np.zeros(n); v_lead = np.zeros(n)
    v[0], gap[0] = set_kmh * KMH, gap0
    collided_at = None

    for k in range(n - 1):
        v_lead[k] = lead_profile(t[k], **lead_kw)
        b_cmd[k] = controller(v[k] / KMH, gap[k], drowsiness)
        # actuator lag
        b_act[k + 1] = b_act[k] + (b_cmd[k] - b_act[k]) * DT / TAU
        # Continuous blend (no on/off chatter):
        #   0 % .. DEAD_ZONE  -> cruise throttle fades out linearly
        #   DEAD_ZONE .. 100 % -> braking deceleration fades in linearly
        b = b_act[k + 1]
        throttle_share = 1.0 - min(b / DEAD_ZONE, 1.0)
        cruise = float(np.clip(K_CRUISE * (set_kmh * KMH - v[k]), -A_CRUISE, A_CRUISE))
        braking = A_MAX * max(0.0, b - DEAD_ZONE) / (100.0 - DEAD_ZONE)
        a = throttle_share * cruise - braking
        acc[k] = a
        v[k + 1] = max(0.0, v[k] + a * DT)
        gap[k + 1] = gap[k] + (v_lead[k] - v[k]) * DT
        if gap[k + 1] <= 0 and collided_at is None:
            collided_at = t[k + 1]
            gap[k + 1:] = 0.0
            v[k + 1:] = 0.0
            break
    v_lead[-1] = lead_profile(t[-1], **lead_kw)
    b_cmd[-1] = b_cmd[-2]
    acc[-1] = acc[-2]

    end = len(t) if collided_at is None else int(round(collided_at / DT))
    jerk = np.diff(acc[:max(end, 2)]) / DT
    return {
        "t": t, "v": v / KMH, "v_lead": v_lead / KMH, "gap": gap,
        "brake": b_cmd, "acc": acc,
        "min_gap": float(gap.min()),
        "collision": collided_at is not None,
        "collided_at": collided_at,
        "peak_decel": float(-acc.min()),
        "rms_jerk": float(np.sqrt(np.mean(jerk ** 2))),
        "brake_switches": int(np.sum(np.abs(np.diff(b_cmd[:max(end, 2)])) > 20)),
    }


def metrics_table(results):
    """results: {label: sim_dict} -> printable table string."""
    head = f"{'Scenario':<28}{'Collision':>10}{'Min gap(m)':>12}{'Peak dec':>10}{'RMS jerk':>10}{'Jumps>20%':>11}"
    lines = [head, "-" * len(head)]
    for label, r in results.items():
        lines.append(f"{label:<28}{('YES' if r['collision'] else 'no'):>10}"
                     f"{r['min_gap']:>12.2f}{r['peak_decel']:>10.2f}"
                     f"{r['rms_jerk']:>10.2f}{r['brake_switches']:>11d}")
    return "\n".join(lines)


SCENARIOS = {
    "Alert driver (0.5)": 0.5,
    "Drowsy driver (5)": 5.0,
    "Sleepy driver (9)": 9.0,
}


def run_all():
    out = {}
    for name, d in SCENARIOS.items():
        out[f"Fuzzy  | {name}"] = simulate(fuzzy_brake, d)
        out[f"Crisp  | {name}"] = simulate(crisp_brake, d)
    return out


if __name__ == "__main__":
    print(metrics_table(run_all()))
