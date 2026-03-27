import numpy as np

class DogLeg:
    def __init__(self, x0, func, max_iterations=100, tolerance=1e-6, bounds=(-1.0, 1.0), max_step=0.2):
        self.func = func
        self.x = np.array(x0, dtype=float, copy=True)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fd_eps = 1e-6
        self.bounds = bounds
        self.max_step = max_step
        self.delta = max_step

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

    def _dogleg_step(self, grad, hessian, delta):
        # Newton step (full trust-region candidate)
        try:
            p_newton = -np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            p_newton = -grad

        if np.linalg.norm(p_newton) <= delta:
            return p_newton

        # Cauchy (steepest descent) point on quadratic model
        gHg = float(grad.T @ hessian @ grad)
        g_norm_sq = float(grad.T @ grad)
        if gHg <= 1e-20:
            p_u = -(delta / max(np.linalg.norm(grad), 1e-12)) * grad
        else:
            p_u = -(g_norm_sq / gHg) * grad

        if np.linalg.norm(p_u) >= delta:
            return (delta / max(np.linalg.norm(p_u), 1e-12)) * p_u

        # Interpolate on the dogleg segment from p_u to p_newton
        p_diff = p_newton - p_u
        a = float(p_diff.T @ p_diff)
        b = 2.0 * float(p_u.T @ p_diff)
        c = float(p_u.T @ p_u) - delta * delta
        disc = max(b * b - 4.0 * a * c, 0.0)
        tau = (-b + np.sqrt(disc)) / (2.0 * max(a, 1e-20))
        tau = np.clip(tau, 0.0, 1.0)
        return p_u + tau * p_diff

    def _model_decrease(self, grad, hessian, step):
        # Predicted reduction of m(0)-m(p) for quadratic model.
        return float(-(grad.T @ step + 0.5 * step.T @ hessian @ step))
    
    def auto_optimize(self, return_history=False):
        history = []
        self.x = self._clip_to_bounds(self.x)

        for _ in range(self.max_iterations):
            grad = self._gradient(self.x)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < self.tolerance:
                break

            hessian = self._hessian(self.x)
            step = self._dogleg_step(grad, hessian, self.delta)

            x_candidate = self._clip_to_bounds(self.x + step)
            step_used = x_candidate - self.x

            f_old = self._value(self.x)
            f_new = self._value(x_candidate)
            actual_reduction = f_old - f_new
            predicted_reduction = self._model_decrease(grad, hessian, step_used)

            if predicted_reduction <= 1e-20:
                rho = -np.inf
            else:
                rho = actual_reduction / predicted_reduction

            # Trust-region radius update
            if rho < 0.25:
                self.delta = max(0.25 * self.delta, 1e-6)
            elif rho > 0.75 and np.linalg.norm(step_used) >= 0.99 * self.delta:
                self.delta = min(2.0 * self.delta, self.max_step)

            # Accept the step if it produced sufficient reduction
            if rho > 0.1:
                self.x = x_candidate

            if return_history:
                history.append(self.x.copy())

        if return_history:
            return history