import skfuzzy as fuzz
import numpy as np
from typing import Literal
from skfuzzy import control as ctrl
from skfuzzy.control.controlsystem import ControlSystemSimulation, ControlSystem

type Action = Literal[0, 1]


def init_simulation() -> tuple[ctrl.Antecedent, ctrl.Antecedent, ctrl.Antecedent, ctrl.Antecedent, ctrl.Consequent, ControlSystem]:
    """
    Method to initialize the simulation.
    Contains static data needed to initialize the simulation.

    :return: Tuple of inputs and outputs as well as control system
    """

    # Sets ranges
    cart_positions_range = np.linspace(-2.4, 2.4, 101)
    cart_velocity_range = np.linspace(-5, 5, 101)
    pole_angles_range = np.linspace(-0.418, 0.418, 101)
    pole_angular_velocity_range = np.linspace(-5, 5, 101)
    force_range = np.linspace(-1, 1, 101)

    # Define fuzzy variables
    cart_pos = ctrl.Antecedent(cart_positions_range, 'cart_position')
    cart_vel = ctrl.Antecedent(cart_velocity_range, 'cart_velocity')
    pole_angle = ctrl.Antecedent(pole_angles_range, 'pole_angle')
    pole_vel = ctrl.Antecedent(pole_angular_velocity_range, 'pole_velocity')
    force = ctrl.Consequent(force_range, 'force')

    # Membership functions
    # cart position (trapezes and triangle)
    cart_pos['left'] = fuzz.trapmf(cart_positions_range, [-2.4, -2.4, -1.5, -0.3])
    cart_pos['center'] = fuzz.trimf(cart_positions_range, [-0.3, 0.0, 0.3])
    cart_pos['right'] = fuzz.trapmf(cart_positions_range, [0.3, 1.5, 2.4, 2.4])

    # cart velocity (triangles)
    cart_vel['neg'] = fuzz.trimf(cart_velocity_range, [-5, -2.5, 0])
    cart_vel['zero'] = fuzz.trimf(cart_velocity_range, [-0.5, 0, 0.5])
    cart_vel['pos'] = fuzz.trimf(cart_velocity_range, [0, 2.5, 5])

    # pole angle (trapezes and triangles)
    pole_angle['NL'] = fuzz.trapmf(pole_angles_range, [-0.418, -0.418, -0.25, -0.08])
    pole_angle['NS'] = fuzz.trimf(pole_angles_range, [-0.15, -0.06, 0.0])
    pole_angle['Z'] = fuzz.trimf(pole_angles_range, [-0.02, 0.0, 0.02])
    pole_angle['PS'] = fuzz.trimf(pole_angles_range, [0.0, 0.06, 0.15])
    pole_angle['PL'] = fuzz.trapmf(pole_angles_range, [0.08, 0.25, 0.418, 0.418])

    # angular velocity
    pole_vel['NL'] = fuzz.trimf(pole_angular_velocity_range, [-10, -5, -1])
    pole_vel['NS'] = fuzz.trimf(pole_angular_velocity_range, [-3, -1.0, 0])
    pole_vel['Z'] = fuzz.trimf(pole_angular_velocity_range, [-0.5, 0.0, 0.5])
    pole_vel['PS'] = fuzz.trimf(pole_angular_velocity_range, [0, 1.0, 3])
    pole_vel['PL'] = fuzz.trimf(pole_angular_velocity_range, [1, 5, 10])

    # output force: left (neg), zero, right (pos)
    force['sleft'] = fuzz.trimf(force_range, [-1.0, -1.0, -0.6])
    force['left'] = fuzz.trimf(force_range, [-0.9, -0.5, 0.0])
    force['zero'] = fuzz.trimf(force_range, [-0.2, 0.0, 0.2])
    force['right'] = fuzz.trimf(force_range, [0.0, 0.5, 0.9])
    force['sright'] = fuzz.trimf(force_range, [0.6, 1.0, 1.0])

    rules = [

        # Recover pole to balance
        # If pole is large tilted and angular velocity pushing it further side -> strong side
        ctrl.Rule(pole_angle['PL'] & pole_vel['PL'], force['sright']),
        ctrl.Rule(pole_angle['NL'] & pole_vel['NL'], force['sleft']),
        # If pole large but angular velocity is smaller or opposite sign -> moderate correction
        ctrl.Rule(pole_angle['PL'] & (pole_vel['PS'] | pole_vel['Z'] | pole_vel['NS']), force['right']),
        ctrl.Rule(pole_angle['NL'] & (pole_vel['NS'] | pole_vel['Z'] | pole_vel['PS']), force['left']),
        # If pole large AND cart is already near the wrong side, move strongly under it
        ctrl.Rule(pole_angle['PL'] & cart_pos['right'], force['sright']),
        ctrl.Rule(pole_angle['NL'] & cart_pos['left'], force['sleft']),
        # If pole large AND cart_vel already pushing away, be aggressive in the correcting direction
        ctrl.Rule(pole_angle['PL'] & cart_vel['pos'], force['sright']),
        ctrl.Rule(pole_angle['NL'] & cart_vel['neg'], force['sleft']),
        # If pole is slightly tilted or velocity is moving the pole, correct the direction
        ctrl.Rule(pole_angle['PS'] | pole_vel['PS'], force['right']),
        ctrl.Rule(pole_angle['NS'] | pole_vel['NS'], force['left']),

        #Centering
        # These fire only when pole is nearly vertical (pole_angle Z and pole_vel Z) — prevents centering while balancing
        # Strong centering if pole is balanced but cart is significantly off-center
        ctrl.Rule(pole_angle['Z'] & pole_vel['Z'] & cart_pos['right'], force['sleft']),
        ctrl.Rule(pole_angle['Z'] & pole_vel['Z'] & cart_pos['left'], force['sright']),

        # Use cart velocity to damp drifting when pole is balanced
        ctrl.Rule(pole_angle['Z'] & pole_vel['Z'] & cart_pos['center'] & cart_vel['pos'], force['left']),
        ctrl.Rule(pole_angle['Z'] & pole_vel['Z'] & cart_pos['center'] & cart_vel['neg'], force['right']),

        # Gentle centering even if pole mildly off but angle small
        ctrl.Rule(pole_angle['Z'] & cart_pos['right'], force['left']),
        ctrl.Rule(pole_angle['Z'] & cart_pos['left'], force['right']),

        # Dampening
        # If pole near-zero but angular velocity non-zero, damp angular velocity
        ctrl.Rule(pole_angle['Z'] & pole_vel['PS'], force['right']),
        ctrl.Rule(pole_angle['Z'] & pole_vel['NS'], force['left']),
        # If pole slightly is tilted but angular velocity large and of opposite sign (returning),
        # choose a moderate action
        ctrl.Rule(pole_angle['PS'] & pole_vel['NL'], force['right']),
        ctrl.Rule(pole_angle['NS'] & pole_vel['PL'], force['left']),

        #Fallback
        ctrl.Rule(pole_angle['Z'] & cart_pos['center'] & pole_vel['Z'] & cart_vel['zero'], force['zero'])
    ]

    # Build control system
    fuzzy_ctrl = ctrl.ControlSystem(rules)

    return cart_pos, cart_vel, pole_angle, pole_vel, force, fuzzy_ctrl


class FuzzyLogic:
    """
    Fuzzy logic implementation.
    """

    def __init__(self):
        self.cart_pos, self.cart_vel, self.pole_angle, self.pole_vel, self.force, self.control = init_simulation()

    def get_sim(self) -> ControlSystemSimulation:
        """
        Creates new control system simulation from configured system.
        :return: ControlSystemSimulation
        """
        return ctrl.ControlSystemSimulation(self.control)

    def view(self):
        """
        Shows the member functions of the simulation.
        """
        self.cart_pos.view()
        self.cart_vel.view()
        self.pole_angle.view()
        self.pole_vel.view()
        self.force.view()
