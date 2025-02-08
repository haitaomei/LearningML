import jax.numpy as jnp
import numpy as np
from safetensors.torch import load_file


class Parameters(object):
    params = {}

    @staticmethod
    def load_weight(weight_path : str = "Llama-3.2-1B-Instruct/model.safetensors"):
        combined_weights = load_file("Llama-3.2-1B-Instruct/model.safetensors")
        for key, value in combined_weights.items():
            v = value.float().numpy()
            Parameters.params[key] = v
        
        # This weight is reused
        if "lm_head.weight" not in Parameters.params:
            Parameters.params["lm_head.weight"] = Parameters.params["model.embed_tokens.weight"]


class PositionalEncodingHelper(object):
    cache = {}
    @staticmethod
    def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config={
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }):
        assert head_dim % 2 == 0, "Embedding dimension must be even"
        inv_freq = 1.0 / (theta_base ** (jnp.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
        # Frequency adjustments
        if freq_config is not None:
            low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
            high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

            wavelen = 2 * jnp.pi / inv_freq

            inv_freq_llama = jnp.where(
                wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
            )

            smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
                freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
            )

            smoothed_inv_freq = (
                (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
            )

            is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
            inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
            inv_freq = inv_freq_llama

        positions = jnp.arange(context_length)
        angles = positions[:, None] * inv_freq[None, :]
        angles = jnp.concatenate([angles, angles], 1)

        cos = jnp.cos(angles)
        sin = jnp.sin(angles)

        return jnp.bfloat16(cos), jnp.bfloat16(sin)

    @staticmethod
    def compute_rope(x, cos, sin):
        assert x.ndim == 3, "Input must be 3D"
        _, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even"

        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]

        cos = jnp.expand_dims(cos[:seq_len, :], axis=0)
        sin = jnp.expand_dims(sin[:seq_len, :], axis=0)

        rotated = jnp.concatenate((-x2, x1), axis=-1)
        x_rotated = (x * cos) + (rotated * sin)

        return x_rotated.astype(x.dtype)
    
    @staticmethod
    def pos_encode(x):
        head_dim = x.shape[-1]
        cos, sin = PositionalEncodingHelper.precompute_rope_params(head_dim)
        if head_dim not in PositionalEncodingHelper.cache:
            PositionalEncodingHelper.cache[head_dim] = [cos, sin]
        
        cos, sin = PositionalEncodingHelper.cache[head_dim]

        return PositionalEncodingHelper.compute_rope(x, cos, sin)
