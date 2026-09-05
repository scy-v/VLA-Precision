from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from vla_precision.acob.training import default_init


def multiple_action_q_function(forward):
    """Vectorize a Q function when each observation carries several actions."""

    def wrapped(self, observations, actions, **kwargs):
        if jnp.ndim(actions) == 3:
            return jax.vmap(
                lambda action: forward(self, observations, action, **kwargs),
                in_axes=1,
                out_axes=-1,
            )(actions)
        return forward(self, observations, actions, **kwargs)

    return wrapped


def ensemblize(module, ensemble_size, out_axes=0):
    """Create independently initialized Flax ensemble members."""
    return nn.vmap(
        module,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=ensemble_size,
    )


class ValueAdvantageCritic(nn.Module):
    """Decompose Q into state value and action-conditioned advantage."""

    encoder: nn.Module | None
    value_network: nn.Module
    advantage_network: nn.Module
    init_final: float | None = None

    @nn.compact
    @multiple_action_q_function
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        train: bool = False,
        return_components: bool = False,
    ) -> jnp.ndarray | dict[str, jnp.ndarray]:
        obs_enc = observations if self.encoder is None else self.encoder(observations)
        head_init = (
            nn.initializers.uniform(-self.init_final, self.init_final)
            if self.init_final is not None
            else default_init()
        )

        value_outputs = self.value_network(obs_enc, train=train)
        value = nn.Dense(1, kernel_init=head_init, name="value_head")(value_outputs)
        value = jnp.squeeze(value, -1)

        advantage_inputs = jnp.concatenate([obs_enc, actions], -1)
        advantage_outputs = self.advantage_network(advantage_inputs, train=train)
        advantage = nn.Dense(1, kernel_init=head_init, name="advantage_head")(advantage_outputs)
        advantage = jnp.squeeze(advantage, -1)

        q = value + advantage
        if return_components:
            return {"q": q, "value": value, "advantage": advantage}
        return q


class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, time):
        half_dim = self.dim // 2
        embeddings = jnp.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings
        return jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)
    
class timeMLP(nn.Module):
    t_dim: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] | str = nn.swish

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        activations = self.activations
        if isinstance(activations, str):
            activations = getattr(nn, activations)

        t = SinusoidalPosEmb(self.t_dim)(t)
        t = nn.Dense(self.t_dim * 2, kernel_init=default_init())(t)
        t = activations(t)
        t = nn.Dense(self.t_dim, kernel_init=default_init())(t)
        
        return t

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] | str = nn.swish
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        activations = self.activations
        if isinstance(activations, str):
            activations = getattr(nn, activations)

        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = activations(x)
        return x


class MLPResNetBlock(nn.Module):
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x


class MLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.swish

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(
                self.hidden_dim,
                act=self.activations,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        x = self.activations(x)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        return x


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value
