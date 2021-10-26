"""

scenario


Author
------
J. Brusey, 24 May 2021


Defines a scenario or start state for simulation.

Revised to use a singleton approach

"""

import pandas as pd
import pkg_resources

scenarios = pd.read_csv(
    pkg_resources.resource_filename(__name__, "model/scenarios.csv")
)


def load_scenarios():
    return scenarios
