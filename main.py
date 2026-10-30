"""
main.py - run this for the viva demo.

    python main.py                 -> worked example + simulation + all plots
    python main.py 90 20 7         -> step-by-step inference for your own inputs
"""

import sys
from braking_system import SYSTEM, RULES, brake, crisp_brake
from fuzzy_engine import defuzz_centroid, defuzz_bisector, defuzz_mom


def explain(s, d, z):
    line = "=" * 70
    print(line)
    print(f" FUZZY BRAKING ASSIST  |  speed={s} km/h   gap={d} m   drowsiness={z}")
    print(line)
    crisp, info = SYSTEM.infer(speed=s, distance=d, drowsiness=z)

    print("\nSTEP 1 - FUZZIFICATION (crisp -> membership degrees)")
    for var, degrees in info["fuzzified"].items():
        txt = "   ".join(f"{t}={m:.2f}" for t, m in degrees.items())
        print(f"  {var:<11}: {txt}")

    print("\nSTEP 2 - RULE EVALUATION (AND = min)  - only rules that fire:")
    for rule, alpha in zip(RULES, info["firing"]):
        if alpha > 0:
            print(f"  α={alpha:.2f}  {rule}")

    print("\nSTEP 3 - IMPLICATION (clip each output set at α, Mamdani min)")
    print("STEP 4 - AGGREGATION (max of all clipped sets)")
    x, agg = brake.universe, info["aggregated"]
    print(f"  aggregated set peak height = {agg.max():.2f}")

    print("\nSTEP 5 - DEFUZZIFICATION")
    print(f"  Centroid (COG)   : {defuzz_centroid(x, agg):6.2f} %   <- used by the system")
    print(f"  Bisector         : {defuzz_bisector(x, agg):6.2f} %")
    print(f"  Mean of Maximum  : {defuzz_mom(x, agg):6.2f} %")
    print(f"\n  >>> FUZZY BRAKE = {crisp:.1f} %     (crisp if-else would give "
          f"{crisp_brake(s, d, z):.0f} %)")
    print(line)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        explain(*map(float, sys.argv[1:]))
        sys.exit()

    explain(70, 18, 6)

    from simulation import run_all, metrics_table
    from plots import make_all
    print("\nSIMULATION - lead car brakes hard at t=4 s (80 -> 10 km/h), start gap 45 m\n")
    table = metrics_table(run_all())
    print(table)
    print("\nGenerating figures ...")
    paths = make_all()
    with open("outputs/metrics.txt", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    for p in paths:
        print("  saved", p)
