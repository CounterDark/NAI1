from dataclasses import dataclass
from typing import Literal
import numpy as np

type Action = Literal[0, 1]


@dataclass
class SimulationState:
    """
    A state of the simulation.
    """

    cart_position: float
    cart_velocity: float
    pole_angle: float
    pole_angular_velocity: float


class CartController:
    """
    A controller for the cart.
    """

    def __init__(self):
        pass

    @classmethod
    def from_observation(
        cls, observation: np.ndarray[tuple[int], np.dtype[np.float32]]
    ) -> SimulationState:
        """
        Convert the observation to a SimulationState object.

        :param observation: The observation from the environment.
        :return state: The SimulationState object.
        """
        return SimulationState(
            cart_position=observation[0],
            cart_velocity=observation[1],
            pole_angle=observation[2],
            pole_angular_velocity=observation[3],
        )

    def make_action(self, state: SimulationState) -> Action:
        """
        Make an action based on the state.

        :param state: The SimulationState object.
        :return action: The action to take.
        """
        return np.random.choice([0, 1])
