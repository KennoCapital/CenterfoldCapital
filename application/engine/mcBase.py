import torch
from products import Portfolio
from models import MCModel


class Sample:
    """
        A sample is a collection of market observations on an event date for the evaluation of the payoff:
            - Numeraire (if the event date is a payment date), and a collection of
            - Forwards
            - Discounts
            - Libors
        fixed on the event date
    """
    def __init__(self):


class Scenario:
    """A scenario is a collection of samples"""


class SampleDef:
    """Definition of what must be sampled"""

class RNG:
    """Random Number Generator"""


def mcSim(
        port:   Portfolio,
        mdl:    Model,
        rng:    RNG,
        N:      int):
    """
    Template algorithm for running Monte Carlo simulation

    :param port:    Portfolio of products to value
    :param mdl:     Model to simulate from
    :param rng:     Random number generator
    :param N:       Number of paths to simulate
    :return:        Payoffs
    """

    # Simulate (over paths)
    mdl.generatePaths():





