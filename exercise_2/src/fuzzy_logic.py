import skfuzzy as fuzz
import numpy as np
from typing import Literal

type Action = Literal[0, 1]


class FuzzyLogic:
    """
    Fuzzy logic implementation.
    """

    def __init__(self):
        pass

    def process(
        self,
        cart_position: float,
        cart_velocity: float,
        pole_angle: float,
        pole_angular_velocity: float,
    ) -> Action:
        cart_positions_range = np.arange(-2.4, 2.4, 0.01)
        pole_angles_range = np.arange(-0.2095, 0.2095, 0.001)
