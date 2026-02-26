from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

# A proof of concept for ragged arrays.
# "Raggedness" can only occur in one (1) axis.


@dataclass(frozen=True)
class RaggedArray:
    lengths: jax.Array
    offsets: jax.Array = field(repr=False)  # Pre-compute for speed
    data: jax.Array
    ragged_axis: int
    stacked_axis: int

    @classmethod
    def from_lists(cls, lists: list[jax.Array], *, ragged_axis, stacked_axis=0) -> RaggedArray:
        # We need to offset the axis since the "stacked axis" doesn't exist
        ragged_element_axis = ragged_axis if ragged_axis <= stacked_axis else ragged_axis - 1
        data = jnp.stack(lists, axis=ragged_element_axis)
        lengths = jnp.asarray(np.fromiter(map(len, lists), dtype=int))
        offsets = jnp.cumsum(lengths)

        return RaggedArray(
            data=data,
            lengths=lengths,
            offsets=offsets,
            ragged_axis=ragged_axis,
        )

    def __getitem__(self, index):
        # I don't know how to get the right index right now, for general case
        a, b, c = index
        b += self.offsets[a]
        return self.data[b, c]


if __name__ == "__main__":
    # An example ragged list we want to push into JAX
    vertices = [
        np.array([[0, 1, 2], [0, 1, 2]]),
        np.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
        np.array([[0, 2, 4], [0, 2, 4]]),
    ]

    leaves = jax.tree.leaves(vertices)
    lengths = jnp.array(jax.tree.map(lambda el: jnp.shape(el)[1], vertices))
    offsets = jnp.cumsum(lengths)
    data = jnp.concatenate(leaves, axis=1)

    ragged = RaggedArray(lengths=lengths, offsets=offsets, data=data)
    print(ragged[1, 0, :])
