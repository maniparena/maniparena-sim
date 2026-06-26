import time


class FixedTimeStampTimer:
    def __init__(self, fixed_dt: float, max_steps: int = 2):
        self._fixed_dt = fixed_dt
        self._max_steps = max_steps
        self._accumulated_time = 0.0
        self._last_time = time.time()

    def step(self, dt: float) -> int:
        self._accumulated_time += dt

        steps = self._accumulated_time // self._fixed_dt
        steps = min(steps, self._max_steps)
        if steps >= 1:
            self._accumulated_time -= steps * self._fixed_dt

        return int(steps)
