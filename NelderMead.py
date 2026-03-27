import numpy as np

class NelderMead:

    """ Nelder-Mead optimization algorithm for 2D functions. 
    Xs is a 2x2 array of the verticies of the simplex in the [[x1, y1], [x2, y2] ..., [xn, yn]]. 
    Zfunc is the function to minimize. """

    def __init__(self, Xs, Zfunc, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, bounds=(-1.0, 1.0)):
        self.bounds = bounds
        self.Xs = self._clip_to_bounds(np.array(Xs, dtype=float, copy=True))
        self.Zfunc = Zfunc
        self.Zs = Zfunc(self.Xs[:, 0], self.Xs[:, 1])
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

    def _clip_to_bounds(self, points):
        lower, upper = self.bounds
        return np.clip(points, lower, upper)

    def step(self):
        # sort the simplex vertices by their function values
        sorted_indices = np.argsort(self.Zs)
        self.Xs = self.Xs[sorted_indices]
        self.Zs = self.Zs[sorted_indices]
        shrink = False
        # find centroid
        centroid = np.mean(self.Xs[:-1], axis=0)
        # reflection 
        reflected = self._clip_to_bounds(centroid + self.alpha * (centroid - self.Xs[-1]))
        reflected_Z = self.Zfunc(reflected[0], reflected[1])

        # reflection is better than the best point -> try expansion
        if reflected_Z < self.Zs[0]:
            expanded = self._clip_to_bounds(centroid + self.gamma * (reflected - centroid))
            expanded_Z = self.Zfunc(expanded[0], expanded[1])
            if expanded_Z < reflected_Z:
                self.Xs[-1] = expanded
                self.Zs[-1] = expanded_Z
            else:
                self.Xs[-1] = reflected
                self.Zs[-1] = reflected_Z

        # reflection is between the best and second-worst -> accept reflection
        elif reflected_Z < self.Zs[-2]:
            self.Xs[-1] = reflected
            self.Zs[-1] = reflected_Z

        # reflection is between second-worst and worst -> outside contraction
        elif reflected_Z < self.Zs[-1]:
            contracted = self._clip_to_bounds(centroid + self.rho * (reflected - centroid))
            contracted_Z = self.Zfunc(contracted[0], contracted[1])
            if contracted_Z <= reflected_Z:
                self.Xs[-1] = contracted
                self.Zs[-1] = contracted_Z
            else:
                shrink = True

        # reflection is worse than worst -> inside contraction
        else:
            contracted = self._clip_to_bounds(centroid + self.rho * (self.Xs[-1] - centroid))
            contracted_Z = self.Zfunc(contracted[0], contracted[1])
            if contracted_Z < self.Zs[-1]:
                self.Xs[-1] = contracted
                self.Zs[-1] = contracted_Z
            else:
                shrink = True

        # shrink
        if shrink:
            self.Xs[1:] = self.Xs[0] + self.sigma * (self.Xs[1:] - self.Xs[0])
            self.Xs = self._clip_to_bounds(self.Xs)
            self.Zs[1:] = self.Zfunc(self.Xs[1:, 0], self.Xs[1:, 1])
    
    def auto_optimize(self, max_iterations=1000, tolerance=1e-6, return_history=False):
        history = []
        for _ in range(max_iterations):
            self.step()
            history.append((self.Xs.copy()))
            if np.std(self.Zs) < tolerance:
                break

        # keep the simplex ordered so index 0 is always the current best point
        sorted_indices = np.argsort(self.Zs)
        self.Xs = self.Xs[sorted_indices]
        self.Zs = self.Zs[sorted_indices]

        if return_history:
            return history