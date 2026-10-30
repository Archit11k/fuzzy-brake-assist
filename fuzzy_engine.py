"""
fuzzy_engine.py
================
A Mamdani fuzzy inference engine written FROM SCRATCH using only NumPy.
(No scikit-fuzzy or any other fuzzy library is used.)

Pipeline implemented here:
    1. Fuzzification      -> crisp input  -> membership degrees
    2. Rule evaluation    -> AND = min, OR = max, NOT = 1 - mu
    3. Implication        -> clip output set at rule strength (Mamdani, min)
    4. Aggregation        -> combine all clipped sets with max (fuzzy union)
    5. Defuzzification    -> centroid / bisector / mean-of-maximum

Course mapping: CSE3121 Lectures 36-38 (fuzzy sets, set-theoretic
operations, membership functions, union / intersection / complement,
fuzzy rules, reasoning and inference).
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. MEMBERSHIP FUNCTIONS
# ---------------------------------------------------------------------------
def trimf(x, a, b, c):
    """Triangular membership function.

            1 |      /\\
              |     /  \\
            0 |____/    \\____
                  a  b   c
    mu(x) = max(min((x-a)/(b-a), (c-x)/(c-b)), 0)
    """
    x = np.asarray(x, dtype=float)
    # a == b or b == c gives a vertical edge (shoulder triangle)
    left = (x >= b).astype(float) if b == a else (x - a) / (b - a)
    right = (x <= b).astype(float) if c == b else (c - x) / (c - b)
    mu = np.minimum(left, right)
    mu = np.where((x < a) | (x > c), 0.0, mu)
    return np.clip(mu, 0.0, 1.0)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function (flat top between b and c).

            1 |     ______
              |    /      \\
            0 |___/        \\___
                 a  b    c  d
    Shoulders are allowed: a == b gives a left shoulder, c == d a right one.
    """
    x = np.asarray(x, dtype=float)
    rise = np.ones_like(x) if b == a else (x - a) / (b - a)
    fall = np.ones_like(x) if d == c else (d - x) / (d - c)
    mu = np.minimum(np.minimum(rise, 1.0), fall)
    # Outside [a, d] membership is exactly zero (handles shoulders correctly)
    mu = np.where((x < a) | (x > d), 0.0, mu)
    return np.clip(mu, 0.0, 1.0)


def gaussmf(x, mean, sigma):
    """Gaussian membership function (included for completeness / comparison)."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


# ---------------------------------------------------------------------------
# 2. SET-THEORETIC OPERATIONS (Zadeh operators)
# ---------------------------------------------------------------------------
def fuzzy_and(*mus):
    """Intersection  A ∩ B  ->  min(muA, muB)"""
    return np.minimum.reduce([np.asarray(m, dtype=float) for m in mus])


def fuzzy_or(*mus):
    """Union  A ∪ B  ->  max(muA, muB)"""
    return np.maximum.reduce([np.asarray(m, dtype=float) for m in mus])


def fuzzy_not(mu):
    """Complement  A'  ->  1 - muA"""
    return 1.0 - np.asarray(mu, dtype=float)


# ---------------------------------------------------------------------------
# 3. LINGUISTIC VARIABLE
# ---------------------------------------------------------------------------
class FuzzyVariable:
    """A linguistic variable, e.g. 'speed' with terms Slow / Medium / Fast."""

    def __init__(self, name, lo, hi, unit="", points=1001):
        self.name = name
        self.unit = unit
        self.universe = np.linspace(lo, hi, points)
        self.terms = {}          # term name -> function(x) -> mu

    def add_term(self, term, mf, *params):
        self.terms[term] = (mf, params)
        return self

    def membership(self, term, x):
        mf, params = self.terms[term]
        return mf(x, *params)

    def fuzzify(self, x):
        """Crisp value -> {term: degree}. Input is clamped to the universe."""
        x = float(np.clip(x, self.universe[0], self.universe[-1]))
        return {t: float(self.membership(t, x)) for t in self.terms}

    def curve(self, term):
        """Membership curve of a term over the whole universe (for plotting)."""
        return self.membership(term, self.universe)


# ---------------------------------------------------------------------------
# 4. RULES
# ---------------------------------------------------------------------------
class Rule:
    """IF <antecedents joined by AND/OR> THEN <output_var> IS <output_term>.

    antecedents : dict {variable_name: term}  e.g. {"speed": "Fast", "distance": "Near"}
                  a term written as "not Near" uses the complement.
    """

    def __init__(self, antecedents, consequent, operator="AND", weight=1.0):
        self.antecedents = antecedents
        self.out_var, self.out_term = consequent
        self.operator = operator.upper()
        self.weight = weight

    def strength(self, fuzzified):
        degrees = []
        for var, term in self.antecedents.items():
            if term.lower().startswith("not "):
                degrees.append(fuzzy_not(fuzzified[var][term[4:]]))
            else:
                degrees.append(fuzzified[var][term])
        combine = fuzzy_and if self.operator == "AND" else fuzzy_or
        return float(combine(*degrees)) * self.weight

    def __str__(self):
        joiner = f" {self.operator} "
        cond = joiner.join(f"{v} is {t}" for v, t in self.antecedents.items())
        return f"IF {cond} THEN {self.out_var} is {self.out_term}"


# ---------------------------------------------------------------------------
# 5. DEFUZZIFICATION
# ---------------------------------------------------------------------------
def defuzz_centroid(x, mu):
    """Centre of gravity:  z* = Σ mu(z)·z / Σ mu(z)"""
    area = np.sum(mu)
    if area == 0:
        return float(x[0])
    return float(np.sum(mu * x) / area)


def defuzz_bisector(x, mu):
    """Vertical line that splits the aggregated area into two equal halves."""
    cum = np.cumsum(mu)
    if cum[-1] == 0:
        return float(x[0])
    return float(x[np.searchsorted(cum, cum[-1] / 2.0)])


def defuzz_mom(x, mu):
    """Mean of Maximum: average of all z where mu(z) is maximal."""
    if mu.max() == 0:
        return float(x[0])
    return float(np.mean(x[np.isclose(mu, mu.max())]))


DEFUZZIFIERS = {
    "centroid": defuzz_centroid,
    "bisector": defuzz_bisector,
    "mom": defuzz_mom,
}


# ---------------------------------------------------------------------------
# 6. MAMDANI INFERENCE SYSTEM
# ---------------------------------------------------------------------------
class MamdaniSystem:
    def __init__(self, inputs, output, rules, defuzz="centroid"):
        self.inputs = {v.name: v for v in inputs}
        self.output = output
        self.rules = rules
        self.defuzz = defuzz

    def infer(self, method=None, **crisp_inputs):
        """Run the full pipeline. Returns (crisp_output, details_dict)."""
        method = method or self.defuzz

        # Step 1 - Fuzzification
        fuzzified = {n: v.fuzzify(crisp_inputs[n]) for n, v in self.inputs.items()}

        # Step 2 + 3 - Rule strength and Mamdani (min) implication
        x = self.output.universe
        aggregated = np.zeros_like(x)
        firing = []
        for rule in self.rules:
            alpha = rule.strength(fuzzified)
            firing.append(alpha)
            if alpha > 0:
                clipped = np.minimum(alpha, self.output.curve(rule.out_term))
                # Step 4 - Aggregation by max (fuzzy union)
                aggregated = np.maximum(aggregated, clipped)

        # Step 5 - Defuzzification
        crisp = DEFUZZIFIERS[method](x, aggregated)

        return crisp, {
            "fuzzified": fuzzified,
            "firing": firing,
            "aggregated": aggregated,
        }

    def compute(self, method=None, **crisp_inputs):
        return self.infer(method=method, **crisp_inputs)[0]
