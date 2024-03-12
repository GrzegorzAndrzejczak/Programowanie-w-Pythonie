# %% [markdown]
# ### RKHS - Reproducing Kernel Hilbert Space Method
# Przestrzeń Hilberta $H\subset C(J)$ z iloczynem skalarnym (inner product) $\langle u,v\rangle$ ma **jądro reprodukujące** $K=\{k_x; \; x\in J\}$ 
# jeśli 
# $$u(x)=\langle u,k_x\rangle\qquad\text{dla wszystkich }\;u\in H, x\in J.$$

# %%
# import abc
from builtins import int
# from typing import Sequence
import numpy as np

from numpy import ndarray


class RK:
    def __init__(self, deg: int, ref: float = 0):
        self._deg = deg
        self._ref = ref

    def Dk(self, p, q: int, x, t: float) -> float:
        n = 2 * self._deg - p - q
        if n <= 0:
            return 0
        if x < 0 and t < 0:
            if (p + q) % 2 == 0:
                return self.Dk(p, q, -x, -t)
            else:
                return -self.Dk(p, q, -x, -t)
        if t < x:
            return self.Dk(q, p, t, x)
        Dk = 1
        if p >= self._deg:  # ±(t-x)^(2deg-p-q-1)/(2deg-p-q-1)!
            if x < 0:
                return 0
            t_x = t - x
            for i in range(1, n):
                Dk = Dk * t_x / i
            if (p - self._deg) % 2 != 0:
                Dk = -Dk
            return Dk
        # now  p < deg,  Dk = 1
        xt, x_t = x * t, 0
        if t > 0:
            x_t = x / t
        if x > 0:
            for i in range(1, self._deg - q):
                Dk = 1 - Dk * x_t * i / (n - i)
            Dk = 1 + Dk * x / (self._deg - p)
        for i in range(1, min(self._deg - p, self._deg - q)):
            Dk = 1 + Dk * xt / (self._deg - p - i) / (self._deg - q - i)
        # and the outer factor
        for i in range(p - q):  # *t^(p-q)/(p-q)!
            Dk = Dk * t / (i + 1)  # if (p>=q)
        for i in range(q - p):  # *x^(q-p)/(q-p)!
            Dk = Dk * x / (i + 1)  # if (p<q)
        return Dk

    def dk(self, p, q: int, x, t: float) -> float:
        return self.Dk(p, q, x - self._ref, t - self._ref)


def main():
    k5 = RK(5)  # k3.dk(2, 1, 0, 1);
    for i in range(10):
        for j in range(10):
            print(f'{i:2d},{j:2d},{k5.dk(2, 1, i * .1, j * .1):7.4f}', sep=': ', end='; ')
        print('\n')

main()



# %%
