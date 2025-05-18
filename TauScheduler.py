import math


class TauScheduler:
    """
    A scheduler for the exponential moving average parameter tau using cosine annealing.

    As described in
    Bootstrap Your Own Latent A New Approach to Self-Supervised Learning (https://arxiv.org/pdf/2006.07733)

    Parameters
    ----------
    num_steps : int
        Total number of steps for the scheduling period
    initial_tau : float
        Initial tau
    """

    def __init__(self, num_steps: int, initial_tau: float) -> None:
        """
        Initialize scheduler with number of steps and starting tau value.
        """
        self.num_steps = num_steps
        self.current_step = 0
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def step(self) -> float:
        """
        Update tau value using cosine scheduler and increment step counter.

        Returns
        -------
        float
            New tau value after update
        """
        new_tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi *
                                                         self.current_step / self.num_steps) + 1) / 2
        self.current_tau = new_tau
        self.current_step += 1
        return new_tau

    def get_current_tau(self) -> float:
        """
        Get current tau value without updating the scheduler.

        Returns
        -------
        float
            Current tau value.
        """
        return self.current_tau

    def __call__(self) -> float:
        """
        Same as step()

        Returns
        -------
        float
            New tau value.
        """
        return self.step()
