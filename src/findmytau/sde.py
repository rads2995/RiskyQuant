import numpy as np
import numpy.typing as npt
import xgboost as xgb

class SDE:
    def __init__(
        self,
        N: int = 10000,
        M: int = 1000,
        mu: float = 0.05,
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
        self.dt = self.T / self.M

        self.X: npt.NDArray[float] = np.zeros((self.M + 1, self.N), dtype=float)
        self.X[0, :] = self.x0

    def simulate(self):

        rng = np.random.default_rng()
        for t in range(self.M):
            dW_X = rng.normal(loc=0.0, scale=np.sqrt(self.dt), size=self.N)
            self.X[t + 1, :] = self.X[t, :] * np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW_X)

    def optimal_stopping(self):

        V = np.maximum(1.0 - self.X[-1, :], 0.0)
        discount: float = np.exp(-0.05 * self.dt)

        for t in range(self.M - 1, -1, -1):
            item = (1.0 - self.X[t, :]) > 0.0
            if not np.any(item):
                continue
            
            features = self.X[t, item].reshape(-1, 1)
            Y = discount * V[item]

            # Quantile Gradient Boosting with XGBoost
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                tree_method='hist',
                quantile_alpha=0.5,
                learning_rate=0.04,
                max_depth = 5,
                device ='cpu'
            )

            model.fit(features, Y)
            C = model.predict(features)           # continuation estimate at quantile tau

            exercise_payoff = np.maximum(1.0 - self.X[t, item], 0.0)
            exercise = exercise_payoff >= C
            V[item] = np.where(exercise, exercise_payoff, V[item])  # stop vs continue

        return np.mean(V)
