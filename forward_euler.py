import scipy.integrate as integrate
import numpy as np

class Forward_Euler(integrate.OdeSolver):

    def __init__(self, fun, t0, y0, t_bound, vectorized, step=0.01, support_complex=False, **extraneous):
        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex)
        self._step = step

    def _step_impl(self):
        try:
            out = self.y + self._step * self.fun(self.t, self.y)
            # Check for NaN
            if False in np.isfinite(out):
                return False, "Output is not finite (NaN or infinity)"
            # Update attributes
            self.t += self._step
            self.y = out
            return True, ""
        except:
            return False, "An unknown error occurred"


    def _dense_output_impl(self):
        # Not sure how this works, but it doesn't seem useful right now
        return integrate.DenseOutput(self.t_old, self.t)
