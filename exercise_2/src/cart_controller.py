import numpy as np

from exercise_2.src.fuzzy_logic import Action, FuzzyLogic


class CartController:
    """
    A controller for the cart.
    """

    def __init__(self):
        self.fuzzy_logic = FuzzyLogic()

    def make_action(
        self,
        cart_position: float,
        cart_velocity: float,
        pole_angle: float,
        pole_angular_velocity: float,
        verbose: bool = False,
    ) -> Action:
        """
        Make an action based on the state.

        :param state: The SimulationState object.
        :return action: The action to take.
        """
        if verbose:
            print(
                f"{cart_position=}, {cart_velocity=}, "
                f"{pole_angle=}, {pole_angular_velocity=}"
            )

        sim = self.fuzzy_logic.get_sim()

        sim.input['cart_position'] = cart_position
        sim.input['cart_velocity'] = cart_velocity
        sim.input['pole_angle'] = pole_angle
        sim.input['pole_velocity'] = pole_angular_velocity

        sim.compute()

        force = sim.output['force']

        if verbose:
            print(f"Force: {force}")

        return np.int64(1 if force > 0 else 0)
