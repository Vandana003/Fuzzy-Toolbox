# toolbox.py

import numpy as np

class FuzzySet:
    def __init__(self, universe, membership_values=None, name="FuzzySet"):
        self.universe = np.array(universe)
        self.membership = np.array(membership_values) if membership_values is not None else np.zeros_like(universe)
        self.name = name

    # === Membership Functions ===
    @staticmethod
    def triangular(universe, a, b, c):
        μ = np.maximum(0, np.minimum((universe - a) / (b - a), (c - universe) / (c - b)))
        return FuzzySet(universe, μ, f"Triangular({a},{b},{c})")

    @staticmethod
    def trapezoidal(universe, a, b, c, d):
        μ = np.maximum(0, np.minimum(np.minimum((universe - a)/(b - a), 1), (d - universe)/(d - c)))
        return FuzzySet(universe, μ, f"Trapezoidal({a},{b},{c},{d})")

    @staticmethod
    def gaussian(universe, c, σ):
        μ = np.exp(-0.5 * ((universe - c) / σ)**2)
        return FuzzySet(universe, μ, f"Gaussian({c},{σ})")

    @staticmethod
    def bell(universe, a, b, c):
        μ = 1 / (1 + np.abs((universe - c) / a)**(2*b))
        return FuzzySet(universe, μ, f"Bell({a},{b},{c})")

    @staticmethod
    def sigmoid(universe, a, c):
        μ = 1 / (1 + np.exp(-a * (universe - c)))
        return FuzzySet(universe, μ, f"Sigmoid({a},{c})")

    @staticmethod
    def manual(universe, values):
        return FuzzySet(universe, values, "Manual")

    # === Fuzzy Set Operations ===
    def __eq__(self, other):  # Equality
        return np.allclose(self.membership, other.membership)

    def complement(self):
        return FuzzySet(self.universe, 1 - self.membership, f"¬({self.name})")

    def intersection(self, other):
        return FuzzySet(self.universe, np.minimum(self.membership, other.membership), f"({self.name}) ∩ ({other.name})")

    def union(self, other):
        return FuzzySet(self.universe, np.maximum(self.membership, other.membership), f"({self.name}) ∪ ({other.name})")

    def algebraic_product(self, other):
        return FuzzySet(self.universe, self.membership * other.membership, f"AlgebraicProd({self.name},{other.name})")

    def multiply_by_crisp(self, k):
        return FuzzySet(self.universe, np.clip(k * self.membership, 0, 1), f"{k}·({self.name})")

    def power(self, p):
        return FuzzySet(self.universe, np.power(self.membership, p), f"({self.name})^{p}")

    def algebraic_sum(self, other):
        μ = self.membership + other.membership - (self.membership * other.membership)
        return FuzzySet(self.universe, μ, f"AlgebraicSum({self.name},{other.name})")

    def algebraic_difference(self, other):
        μ = self.membership - self.membership * other.membership
        return FuzzySet(self.universe, μ, f"AlgDiff({self.name},{other.name})")

    def bounded_sum(self, other):
        μ = np.minimum(1, self.membership + other.membership)
        return FuzzySet(self.universe, μ, f"BoundedSum({self.name},{other.name})")

    def bounded_difference(self, other):
        μ = np.maximum(0, self.membership - other.membership)
        return FuzzySet(self.universe, μ, f"BoundedDiff({self.name},{other.name})")

    # === Implications ===
    def zadeh_implication(self, other):
        μ = np.maximum(1 - self.membership, other.membership)
        return FuzzySet(self.universe, μ, f"ZadehImp({self.name},{other.name})")

    def mamdani_implication(self, other):
        μ = np.minimum(self.membership, other.membership)
        return FuzzySet(self.universe, μ, f"MamdaniImp({self.name},{other.name})")

    def larsen_implication(self, other):
        μ = self.membership * other.membership
        return FuzzySet(self.universe, μ, f"LarsenImp({self.name},{other.name})")

    # === Defuzzification Methods ===
    def centroid(self):
        return np.sum(self.universe * self.membership) / np.sum(self.membership)

    def bisector(self):
        area = np.sum(self.membership)
        cumulative = np.cumsum(self.membership)
        idx = np.searchsorted(cumulative, area / 2)
        return self.universe[idx]

    def mean_of_maximum(self):
        maxμ = np.max(self.membership)
        return np.mean(self.universe[self.membership == maxμ])

    def smallest_of_maximum(self):
        maxμ = np.max(self.membership)
        return np.min(self.universe[self.membership == maxμ])

    def largest_of_maximum(self):
        maxμ = np.max(self.membership)
        return np.max(self.universe[self.membership == maxμ])

    def lambda_cut(self, α=0.5):
        return self.universe[self.membership >= α]

    def weighted_average(self):
        return np.sum(self.universe * self.membership) / np.sum(self.membership)

    def height_method(self):
        return np.sum(self.universe * np.max(self.membership)) / np.sum(np.max(self.membership))

    def center_of_sums(self):
        return self.centroid()  # Same result mathematically

    def center_of_area(self):
        return self.centroid()  # Alias
