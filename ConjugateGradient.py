import numpy as np

class ConjugateGradient:
    def __init__(self, x0, func, max_iterations=100, tolerance=1e-6, bounds=(-1.0, 1.0), max_step=0.2):
        self.func = func
        self.x = np.array(x0, dtype=float, copy=True)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fd_eps = 1e-6
        self.bounds = bounds
        self.max_step = max_step

    def _clip_to_bounds(self, x):
        lower, upper = self.bounds
        return np.clip(x, lower, upper)

    def _value(self, x):
        x_clipped = self._clip_to_bounds(x)
        return float(self.func(x_clipped[0], x_clipped[1]))

    def _gradient(self, x):
        # Central finite differences for a plain callable f(x, y).
        h = self.fd_eps
        grad = np.zeros_like(x, dtype=float)

        x_center = self._clip_to_bounds(x.copy())

        x_forward = x_center.copy()
        x_backward = x_center.copy()
        x_forward[0] += h
        x_backward[0] -= h
        grad[0] = (self._value(x_forward) - self._value(x_backward)) / (2.0 * h)

        y_forward = x_center.copy()
        y_backward = x_center.copy()
        y_forward[1] += h
        y_backward[1] -= h
        grad[1] = (self._value(y_forward) - self._value(y_backward)) / (2.0 * h)

        return grad

    def _hessian(self, x):
        # Central finite differences for the 2x2 Hessian matrix.
        h = self.fd_eps
        x_center = self._clip_to_bounds(x.copy())

        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])

        f0 = self._value(x_center)
        f_xph = self._value(x_center + h * e1)
        f_xmh = self._value(x_center - h * e1)
        f_yph = self._value(x_center + h * e2)
        f_ymh = self._value(x_center - h * e2)

        h_xx = (f_xph - 2.0 * f0 + f_xmh) / (h * h)
        h_yy = (f_yph - 2.0 * f0 + f_ymh) / (h * h)

        f_pp = self._value(x_center + h * e1 + h * e2)
        f_pm = self._value(x_center + h * e1 - h * e2)
        f_mp = self._value(x_center - h * e1 + h * e2)
        f_mm = self._value(x_center - h * e1 - h * e2)
        h_xy = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)

        return np.array([[h_xx, h_xy], [h_xy, h_yy]], dtype=float)

    def _step_size_from_hessian(self, x, p, grad):
        # Quadratic-model step length: alpha = -(g^T p) / (p^T H p).
        hessian = self._hessian(x)
        numerator = -np.dot(grad, p)
        denominator = np.dot(p, hessian.dot(p))
        alpha = 1e-3
        if denominator > 1e-20:
            candidate = numerator / denominator
            if np.isfinite(candidate) and candidate > 0.0:
                alpha = candidate

        # Trust-region style cap on actual displacement length (always applied).
        p_norm = np.linalg.norm(p)
        if p_norm > 1e-12:
            alpha = min(alpha, self.max_step / p_norm)

        return min(alpha, 1.0)

    def auto_optimize(self, return_history=False):
        history = []
        self.x = self._clip_to_bounds(self.x)
        grad = self._gradient(self.x)
        p = -grad

        for iteration in range(self.max_iterations):
            # Reset direction if it is not a descent direction.
            if np.dot(grad, p) >= 0.0 or iteration % len(self.x) == 0:
                p = -grad

            x_prev = self.x.copy()
            f_prev = self._value(x_prev)
            alpha = self._step_size_from_hessian(self.x, p, grad)
            self.x = self._clip_to_bounds(self.x + alpha * p)
            grad_new = self._gradient(self.x)

            # Reject non-improving step and fallback to a short steepest-descent move.
            if self._value(self.x) > f_prev:
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-12:
                    fallback_alpha = min(1e-3, self.max_step / grad_norm)
                else:
                    fallback_alpha = 0.0
                self.x = self._clip_to_bounds(x_prev + fallback_alpha * (-grad))
                grad_new = self._gradient(self.x)
                p = -grad_new

            if return_history:
                history.append(self.x.copy())

            if np.linalg.norm(grad_new) < self.tolerance:
                break

            # Polak-Ribiere+ with clipping to avoid explosive directions.
            beta_den = max(np.dot(grad, grad), 1e-20)
            beta_pr = np.dot(grad_new, grad_new - grad) / beta_den
            beta = np.clip(beta_pr, 0.0, 1.0)
            p = -grad_new + beta * p
            grad = grad_new

        if return_history:
            return history