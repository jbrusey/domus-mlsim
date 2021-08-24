"""

scenario


Author
------
J. Brusey, 24 May 2021


Defines a scenario or start state for simulation.



"""

import pandas as pd
import pkg_resources


def load_scenarios():
    return pd.read_csv(pkg_resources.resource_filename(__name__, "model/scenarios.csv"))
