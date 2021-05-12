"""

run_model


Author
------
J. Brusey, 4-March-2021


Simulator that loads a ML model and single steps simulation.

"""


import numpy as np
from .partial_scaler import PartialScaler


class MLSim:
    """
    Discrete time simulator derived from machine-learnt model.


    """
    def __init__(self,
                 model,
                 scaler,
                 initial_state,
                 xlag,
                 ulag,
                 xlen,
                 ulen,
                 interval=1,
                 initial_clock=0
                 ):
        """construct a simulator object

        Parameters
        ----------
        model : object

          model object with predict function

        scaler : object

          sklearn scaler object used with model to scale x, u vector

        initial_state : array-like

          vector containing initial state. Can also be a 2D array of
          xlag vectors for more complete initialisation. If only a
          single vector is supplied, that vector is replicated xlag
          times to make up the initial state.

        xlag : integer

          number of lagged state values. This must match the
          parameters used to develop the model

        ulag : integer

          number of lagged control values. This must match the
          parameters used to develop the model

        xlen : integer

          length of the state vector

        ulen : integer

          length of the control vector

        interval : integer

          time interval that the simulator was developed using, in
          seconds

        initial_clock : integer

          initial clock value so that first step will have time
          initial_clock + interval

        """
        self.model = model
        if initial_state.shape[0] == 1:
            self.xt = np.repeat(initial_state.reshape(1, -1),
                                xlag,
                                axis=0)
        else:
            self.xt = initial_state.reshape(xlag, xlen)
        self.xlag = xlag
        self.ulag = ulag
        self.xlen = xlen
        self.ulen = ulen
        self.interval = interval
        self.clock = initial_clock
        self.first = True
        if scaler is not None:
            self.x_scaler = PartialScaler(scaler, 0, xlen, xlen + ulen)
            self.u_scaler = PartialScaler(scaler, xlen, xlen + ulen, xlen + ulen)
            self.xt = self.x_scaler.transform(self.xt)
        else:
            self.x_scaler = self.u_scaler = None

    def step(self, ut):
        """simulate a single time step

        Given the current state history given by ~xt~ and current
        control history ~ut~

        Parameters
        ----------

        ut : array-like

          control input for current time step

        Returns
        -------

        (time, state vector at end of time step)

        """
        ut = np.array(ut).reshape(1, -1)
        ut = self.u_scaler.transform(ut)
        assert self.xt.shape[0] == self.xlag
        assert self.xt.shape[1] == self.xlen
        if self.first:
            # for first step, assume that the same control inputs were
            # used for all previous steps
            self.ut = np.repeat(ut,
                                self.ulag,
                                axis=0)
        else:
            self.ut = np.append(self.ut[1:],
                                ut,
                                axis=0)

        x = np.hstack((self.xt.reshape(1, -1), self.ut.reshape(1, -1)))
        new_xt = self.model.predict(x)
        self.xt = np.append(self.xt[1:],
                            new_xt.reshape(1, -1),
                            axis=0)
        self.clock += self.interval
        return (self.clock, self.x_scaler.inverse_transform(new_xt))
