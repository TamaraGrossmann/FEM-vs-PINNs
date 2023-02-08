import flax
from typing import Sequence

class PDESolution(flax.linen.Module):
    features: Sequence[int]

    @flax.linen.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = flax.linen.activation.tanh(flax.linen.Dense(feat)(x)) 
        x = flax.linen.Dense(self.features[-1],kernel_init=flax.linen.initializers.glorot_uniform())(x)
        return x