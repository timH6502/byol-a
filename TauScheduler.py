import math


class TauScheduler:

    def __init__(self, num_steps: int, initial_tau: float) -> None:
        self.num_steps = num_steps
        self.current_step = 0
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def step(self) -> float:
        new_tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi *
                                                         self.current_step / self.num_steps) + 1) / 2
        self.current_tau = new_tau
        self.current_step += 1
        return new_tau

    def get_current_tau(self) -> float:
        return self.current_tau

    def __call__(self) -> float:
        return self.step()
