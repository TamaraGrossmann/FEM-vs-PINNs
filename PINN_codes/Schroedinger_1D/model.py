import flax
from typing import Sequence

class PDESolution(flax.linen.Module):
    features: Sequence[int]

    @flax.linen.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = flax.linen.tanh(flax.linen.Dense(feat)(x))
        x = flax.linen.Dense(self.features[-1])(x)
        return x