# Viva Preparation: Fuzzy Braking Assist

## 60-second pitch (say this first)
> "We built a fuzzy braking-assist system that takes the car's speed, the gap to the car ahead and the driver's drowsiness level, and outputs a braking force from 0 to 100 %. We wrote the whole Mamdani fuzzy inference engine ourselves in NumPy: membership functions, min/max operators, 27 rules, clipping, max aggregation and centroid defuzzification. We then simulated a sudden-braking scenario and compared it to a normal if-else controller. The fuzzy system avoided the crash in all three driver states and braked 2–6 times more smoothly, while the if-else one crashed because its output jumps in steps."

---

## A. Fuzzy logic basics

**1. What is fuzzy logic? How is it different from classical (crisp) logic?**
Classical logic: an element is either in a set (1) or not (0). Fuzzy logic allows *partial membership* μ ∈ [0, 1]. Example: 70 km/h is Medium with degree 0.67 instead of "Medium = yes/no".

**2. Who introduced fuzzy sets?** Lotfi A. Zadeh, 1965.

**3. What is a membership function?** A function μ_A(x): X → [0, 1] giving the degree to which x belongs to fuzzy set A.

**4. Which membership functions did you use and why?**
Triangular (`trimf`) for "middle" terms and trapezoidal (`trapmf`) for the extreme/shoulder terms (Slow, Fast, Near, Far…). They are cheap to compute, easy to explain and tune, and standard in control systems. Gaussian is also implemented in the engine but not used, because it never reaches exactly 0.

**5. Triangular MF formula?** μ(x) = max(min((x−a)/(b−a), (c−x)/(c−b)), 0).

**6. Trapezoidal MF formula?** μ(x) = max(min((x−a)/(b−a), 1, (d−x)/(d−c)), 0).

**7. What is a linguistic variable?** A variable whose values are words. Here "speed" takes the values {Slow, Medium, Fast}, and each word is a fuzzy set.

**8. Universe of discourse?** The full range of a variable: speed 0–120, gap 0–100, drowsiness 0–10, brake 0–100.

**9. Support, core, height of a fuzzy set?**
Support = {x | μ > 0}; core = {x | μ = 1}; height = max μ. For Medium speed (30, 60, 90): support (30, 90), core {60}, height 1.

**10. Is fuzzy membership the same as probability?** No. Probability measures the *likelihood* that an event occurs. Membership measures the *degree* to which something has a property. Memberships of different sets need not sum to 1. (Ours do at most points, but that's a design choice, not a rule.)

## B. Set-theoretic operations (syllabus: union, intersection, complement)

**11. Union, intersection, complement?** Union: max(μA, μB). Intersection: min(μA, μB). Complement: 1 − μA. These are the Zadeh operators, implemented in `fuzzy_and`, `fuzzy_or` and `fuzzy_not`.

**12. Why min for AND?** A rule is only as true as its weakest condition. Other T-norms exist: algebraic product (a·b), Łukasiewicz (max(0, a+b−1)).

**13. Do the excluded middle and contradiction laws hold?** No. A ∪ A' ≠ X and A ∩ A' ≠ ∅. Example: μ = 0.4 → max(0.4, 0.6) = 0.6 ≠ 1. This is a key difference from crisp sets.

## C. Inference

**14. Steps of a Mamdani FIS?** Fuzzification → rule evaluation (firing strength) → implication (clip) → aggregation (max) → defuzzification.

**15. Mamdani vs Sugeno?** In Mamdani the outputs are fuzzy sets and you defuzzify; it is intuitive and interpretable. In Sugeno the outputs are constants or linear functions, and you take a weighted average. Sugeno is faster and suits optimisation and ANFIS. We chose Mamdani because the rules read like human expert knowledge ("brake is Hard").

**16. What is firing strength α?** The degree to which a rule's IF-part is satisfied: α = min of the antecedent memberships.

**17. What is implication (clipping)?** Output set μ'(z) = min(α, μ_out(z)). The output set is cut at height α. (Larsen implication would *scale* instead: α·μ_out(z).)

**18. Aggregation?** Combine all rules' clipped output sets with max into one fuzzy set.

**19. How many rules, and why 27?** 3 speed × 3 distance × 3 drowsiness = 27. That is a *complete* rule base: every combination of terms has a rule, so no input leaves the system with no output. The unit test `test_rule_base_complete` checks this over the whole input grid.

**20. How did you decide the rules?** From driving-safety knowledge. More speed, less gap or more drowsiness means more braking. Drowsiness escalates the braking level because the driver reacts later. A sleepy driver gets braking even at a Far gap, so the car is slowed to a safe stop.

**21. Walk me through your example (70 km/h, 18 m, 6).**
Speed Medium = (90−70)/30 = 0.67. Gap Near = (20−18)/12 = 0.17 and Medium = (18−10)/15 = 0.53. Drowsy = (8−6)/3 = 0.67.
Rule (Medium, Near, Drowsy → Emergency): α = min(0.67, 0.17, 0.67) = 0.17.
Rule (Medium, Medium, Drowsy → Hard): α = 0.53.
Clip Emergency at 0.17 and Hard at 0.53, take the max, then the centroid gives 76.4 %.

## D. Defuzzification

**22. What is defuzzification?** Converting the aggregated fuzzy output into one crisp number that the actuator can use.

**23. Methods you implemented?** Centroid (centre of gravity) z* = Σμ(z)z / Σμ(z); Bisector (splits the area in half); Mean of Maximum (average of the z values where μ is highest).

**24. Why centroid?** It considers the whole shape, so the output changes smoothly as the inputs change. That matters for comfortable braking. MOM can jump suddenly when a different rule becomes the strongest (see figure 7).

**25. Why does the brake never reach exactly 100 % or 0 %?** The centroid of a set always lies inside it, away from the edges. The Emergency set's centroid is ≈ 92 % and None's is ≈ 3 %. That's why the simulation has a 5 % **dead zone**: below it there is no braking (like brake-pedal free play). Fixes: extend the output sets beyond the universe, or use Sugeno.

## E. Design and results

**26. Why fuzzy for braking and not if-else?** Driving decisions are naturally vague ("a bit close", "quite fast"). If-else has hard thresholds, so a 0.1 m change in gap can switch the brake from 0 % to 75 %. That produces jerky braking and late reactions. Fuzzy gives a smooth, continuous control surface (figures 3 and 4).

**27. What did the simulation show?** The lead car brakes 80 → 10 km/h at 7 m/s². The fuzzy controller avoided a collision in all 3 driver states (min gap 5.0 / 13.2 / 45 m) with RMS jerk 1.0–2.2 m/s³. The crisp controller crashed in all 3, with 2–6× higher jerk.

**28. Isn't the crisp controller just badly tuned?** Partly. A better crisp controller might avoid the crash. But it would still change in steps, and every extra situation needs more thresholds. Fuzzy captures the same expert knowledge in 27 readable rules and blends them automatically.

**29. What is jerk and why measure it?** Jerk is the rate of change of acceleration (m/s³). High jerk means passenger discomfort and a greater chance of wheel lock or skidding.

**30. How is drowsiness measured in real cars?** PERCLOS (the percentage of time the eyes are closed), eye-aspect-ratio from a camera, steering-correction patterns, or lane deviation. Our 0–10 score stands in for that sensor.

**31. Limitations?** No relative-speed input, so the car keeps braking at a standstill even when the gap is growing. The MFs are hand-tuned. The physics is a simple point-mass model with no road-friction input.

**32. How would you improve it?** Add time-to-collision or closing speed as an input. Optimise the MF parameters with a **Genetic Algorithm** (that links to CO5). Use **ANFIS** to learn them from driving data. Add road condition as an input.

**33. Which course outcome does this map to?** CO4: "Apply soft computing and fuzzy techniques to solve problems and improve entrepreneurship skills". The application to ADAS (driver assistance) is a real commercial product area.

**34. Real-world uses of fuzzy control?** Washing machines, AC units, camera autofocus, the Sendai subway (Japan, 1987), ABS and cruise control, cement kilns.

## F. Code questions they might ask you to show

- *Where's the membership function?* → `fuzzy_engine.py`: `trimf` and `trapmf`
- *Where's AND?* → `Rule.strength()` calls `fuzzy_and` (= `np.minimum`)
- *Where's clipping and aggregation?* → `MamdaniSystem.infer()`: `np.minimum(alpha, curve)` and `np.maximum(aggregated, clipped)`
- *Where's the centroid?* → `defuzz_centroid`: `np.sum(mu * x) / np.sum(mu)`
- *Change a rule live* → edit `RULE_TABLE` in `braking_system.py` and rerun `python main.py 70 18 6`
- *Did you use a library?* → No. Only NumPy for arrays and Matplotlib for plots. `skfuzzy` is not used anywhere.

## Demo order for the viva (5 minutes)
1. `streamlit run app.py`: move the sliders, show the brake value changing smoothly.
2. Slide drowsiness 0 → 10 at a fixed speed and gap: braking rises.
3. Open the "Rules fired" tab: show α = min(...).
4. Open the "Simulation" tab: fuzzy is safe, crisp crashes.
5. Show figure 3 (control surface) and figure 4 (smooth vs step).
6. Run `python main.py 70 18 6` and explain the 5 printed steps.
