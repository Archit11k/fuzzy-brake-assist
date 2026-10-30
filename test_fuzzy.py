"""test_fuzzy.py - sanity checks. Run:  python test_fuzzy.py"""

import numpy as np
from fuzzy_engine import trimf, trapmf, fuzzy_and, fuzzy_or, fuzzy_not, defuzz_centroid
from braking_system import fuzzy_brake, RULES, speed, distance

def test_membership():
    assert trimf(5, 0, 5, 10) == 1.0
    assert trimf(0, 0, 5, 10) == 0.0
    assert abs(trimf(2.5, 0, 5, 10) - 0.5) < 1e-9
    assert trapmf(0, 0, 0, 10, 20) == 1.0          # left shoulder
    assert abs(trapmf(15, 0, 0, 10, 20) - 0.5) < 1e-9

def test_operators():
    assert fuzzy_and(0.3, 0.7) == 0.3
    assert fuzzy_or(0.3, 0.7) == 0.7
    assert abs(fuzzy_not(0.3) - 0.7) < 1e-9

def test_centroid_symmetric():
    x = np.linspace(0, 10, 1001)
    assert abs(defuzz_centroid(x, trimf(x, 2, 5, 8)) - 5) < 1e-6

def test_rule_base_complete():
    assert len(RULES) == 27
    # every point of the input space fires at least one rule (no "holes")
    from braking_system import SYSTEM
    for s in np.linspace(0, 120, 25):
        for d in np.linspace(0, 100, 25):
            for z in np.linspace(0, 10, 11):
                _, info = SYSTEM.infer(speed=s, distance=d, drowsiness=z)
                assert max(info["firing"]) > 0, (s, d, z)

def test_monotonic_behaviour():
    # closer -> more braking ; faster -> more braking ; sleepier -> more braking
    assert fuzzy_brake(80, 10, 1) > fuzzy_brake(80, 30, 1) > fuzzy_brake(80, 60, 1)
    assert fuzzy_brake(110, 25, 1) > fuzzy_brake(40, 25, 1)
    assert fuzzy_brake(70, 25, 9) > fuzzy_brake(70, 25, 1)

def test_no_brake_when_safe():
    assert fuzzy_brake(60, 90, 0) < 5      # inside dead zone

if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"\nAll {len(tests)} tests passed.")
