
from .sde import SDE

def main():
    sde: SDE = SDE()
    sde.simulate()
    print(sde.optimal_stopping())
