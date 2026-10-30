"""
braking_system.py
==================
Fuzzy Driver-Drowsiness & Braking Assist controller.

INPUTS
  speed       : own vehicle speed               0 - 120 km/h
  distance    : gap to the vehicle ahead        0 - 100 m
  drowsiness  : driver drowsiness score         0 - 10
                (0 = fully alert, 10 = eyes closing / micro-sleep;
                 in a real car this comes from an eye-closure (PERCLOS)
                 camera or steering-pattern sensor)

OUTPUT
  brake       : braking force                   0 - 100 %

IDEA
  A faster car needs more braking, a closer car needs more braking, and a
  drowsy driver reacts late - so the assist must brake EARLIER and HARDER
  when drowsiness is high.
"""

from fuzzy_engine import FuzzyVariable, Rule, MamdaniSystem, trimf, trapmf

# ---------------------------------------------------------------------------
# Linguistic variables and membership functions
# ---------------------------------------------------------------------------
speed = (FuzzyVariable("speed", 0, 120, "km/h")
         .add_term("Slow",   trapmf, 0, 0, 20, 50)
         .add_term("Medium", trimf, 30, 60, 90)
         .add_term("Fast",   trapmf, 70, 100, 120, 120))

distance = (FuzzyVariable("distance", 0, 100, "m")
            .add_term("Near",   trapmf, 0, 0, 8, 20)
            .add_term("Medium", trimf, 10, 25, 40)
            .add_term("Far",    trapmf, 30, 50, 100, 100))

drowsiness = (FuzzyVariable("drowsiness", 0, 10, "score")
              .add_term("Alert",  trapmf, 0, 0, 2, 4)
              .add_term("Drowsy", trimf, 2, 5, 8)
              .add_term("Sleepy", trapmf, 6, 8, 10, 10))

brake = (FuzzyVariable("brake", 0, 100, "%")
         .add_term("None",      trimf, 0, 0, 10)
         .add_term("Light",     trimf, 5, 25, 45)
         .add_term("Moderate",  trimf, 30, 50, 70)
         .add_term("Hard",      trimf, 55, 75, 95)
         .add_term("Emergency", trapmf, 80, 92, 100, 100))

# ---------------------------------------------------------------------------
# Rule base: 3 x 3 x 3 = 27 rules (complete - every input combination covered)
#
#   ALERT  : base safety logic (brake only when the gap is unsafe)
#   DROWSY : the driver reacts ~1 s later, so the assist escalates
#            Near/Medium-gap braking by one level
#   SLEEPY : escalates further AND gently slows the car even when the gap
#            is Far (a sleeping driver should not keep cruising)
# ---------------------------------------------------------------------------
RULE_TABLE = {
    #            distance:  Near         Medium       Far
    "Alert": {
        "Slow":   ["Hard",      "Light",     "None"],
        "Medium": ["Emergency", "Moderate",  "None"],
        "Fast":   ["Emergency", "Hard",      "None"],
    },
    "Drowsy": {
        "Slow":   ["Emergency", "Moderate",  "None"],
        "Medium": ["Emergency", "Hard",      "None"],
        "Fast":   ["Emergency", "Emergency", "Light"],
    },
    "Sleepy": {
        "Slow":   ["Emergency", "Hard",      "Light"],
        "Medium": ["Emergency", "Emergency", "Light"],
        "Fast":   ["Emergency", "Emergency", "Moderate"],
    },
}

DIST_TERMS = ["Near", "Medium", "Far"]


def build_rules():
    rules = []
    for drowsy_term, rows in RULE_TABLE.items():
        for speed_term, outputs in rows.items():
            for dist_term, out_term in zip(DIST_TERMS, outputs):
                rules.append(Rule(
                    {"speed": speed_term, "distance": dist_term,
                     "drowsiness": drowsy_term},
                    ("brake", out_term),
                    operator="AND"))
    return rules


RULES = build_rules()
SYSTEM = MamdaniSystem([speed, distance, drowsiness], brake, RULES,
                       defuzz="centroid")


def fuzzy_brake(speed_kmh, distance_m, drowsiness_score, method="centroid"):
    """Convenience wrapper -> braking force in %."""
    return SYSTEM.compute(method=method, speed=speed_kmh,
                          distance=distance_m, drowsiness=drowsiness_score)


# ---------------------------------------------------------------------------
# Baseline: conventional CRISP (if-else threshold) controller for comparison
# ---------------------------------------------------------------------------
def crisp_brake(speed_kmh, distance_m, drowsiness_score):
    """Hard-threshold controller - what you'd write WITHOUT fuzzy logic.
    Output jumps in steps; tiny input changes near a threshold cause big
    output changes (jerky braking)."""
    if distance_m < 8:
        b = 100
    elif distance_m < 20 and speed_kmh > 60:
        b = 75
    elif distance_m < 20:
        b = 50
    elif distance_m < 35 and speed_kmh > 80:
        b = 40
    else:
        b = 0
    if drowsiness_score > 6 and b > 0:
        b = min(100, b + 30)
    return float(b)


if __name__ == "__main__":
    print(f"{len(RULES)} rules loaded. Sample rules:")
    for r in RULES[:5]:
        print("  ", r)
    for args in [(30, 80, 1), (60, 40, 1), (100, 20, 1), (100, 20, 9)]:
        print(f"speed={args[0]:>3} km/h  dist={args[1]:>3} m  drowsy={args[2]}  "
              f"->  fuzzy={fuzzy_brake(*args):5.1f}%   crisp={crisp_brake(*args):5.1f}%")
