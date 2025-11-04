import numpy as np

from exercise_2.src.fuzzy_logic import Action


class CartController:
    """
    A controller for the cart.
    """

    def __init__(self):
        pass

    def make_action(
        self,
        cart_position: float,
        cart_velocity: float,
        pole_angle: float,
        pole_angular_velocity: float,
    ) -> Action:
        """
        Make an action based on the state.

        :param state: The SimulationState object.
        :return action: The action to take.
        """
        print(
            f"{cart_position=}, {cart_velocity=}, "
            f"{pole_angle=}, {pole_angular_velocity=}"
        )
        return np.int64(1)
