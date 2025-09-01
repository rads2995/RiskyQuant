import numpy as np
import numpy.typing as npt

from sklearn.ensemble import HistGradientBoostingRegressor

class SDE:
    def __init__(
        self,
        N: int = 10,
        M: int = 1000,
        mu: float = 1.0,
        sigma: float = 0.2,
        x0: float = 1.0,
        T: float = 1.0
    ):
        self.N: int = N
        self.M: int = M
        
        self.mu: float = mu
        self.sigma: float = sigma
        self.x0: float = x0

        self.T: float = T

        self.X: npt.NDArray[float] = np.zeros((self.M + 1, self.N), dtype=float)
        self.X[0, :] = self.x0

    def simulate(self):

        dt = self.T / self.M

        rng = np.random.default_rng()
        for t in range(self.M):
            dW_X = rng.normal(loc=0.0, scale=np.sqrt(dt), size=self.N)
            self.X[t + 1, :] = self.X[t, :] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW_X)

        t = np.linspace(0.0, self.T, self.M + 1)

    def optimal_stopping(self):

        V = np.maximum(1.5 - self.X[-1, :], 0.0)

        for t in range(self.M - 1, -1, -1):
            item = (1.5 - self.X[t, :]) > 0.0
            if not np.any(item):
                continue
            
            features = self.X[t, item].reshape(-1, 1)
            Y = V[item]

            # Quantile Gradient Boosting
            model = HistGradientBoostingRegressor(
                loss='quantile', quantile=0.5
            )
            model.fit(features, Y)
            C = model.predict(features)           # continuation estimate at quantile tau

            exercise_payoff = np.maximum(1.5 - self.X[t, item], 0.0)
            exercise = exercise_payoff >= C
            V[item] = np.where(exercise, exercise_payoff, V[item])  # stop vs continue

        return np.mean(V)
