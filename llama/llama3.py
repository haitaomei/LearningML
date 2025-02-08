import jax
import jax.numpy as jnp
import numpy as np

from utils import Parameters, PositionalEncodingHelper

dim = 2048
n_heads = 32
n_kv_heads = 8
context_length = 4096
n_layers = 16

def generate(input_ids, n_layers = 16):
    input_ids = np.asarray(input_ids, dtype=np.int32)
    tok_embedding = Parameters.params['model.embed_tokens.weight']
    x = tok_embedding[input_ids]

    for i in range(n_layers):
        x = transformer_block(x, i)

    rms_out_w = Parameters.params['model.norm.weight']
    x = rms_norm(x, rms_out_w)
    
    linear_out_weight = Parameters.params['lm_head.weight'].T
    logit = linear(x, linear_out_weight)
    return logit

def transformer_block(x, l, ):
    rms_w1 = Parameters.params[f"model.layers.{l}.input_layernorm.weight"]
    rms_w2 = Parameters.params[f"model.layers.{l}.post_attention_layernorm.weight"]
    gate_proj = Parameters.params[f"model.layers.{l}.mlp.gate_proj.weight"].T
    up_proj = Parameters.params[f"model.layers.{l}.mlp.up_proj.weight"].T
    down_proj = Parameters.params[f"model.layers.{l}.mlp.down_proj.weight"].T
    q = Parameters.params[f"model.layers.{l}.self_attn.q_proj.weight"].T
    k = Parameters.params[f"model.layers.{l}.self_attn.k_proj.weight"].T
    v = Parameters.params[f"model.layers.{l}.self_attn.v_proj.weight"].T
    o = Parameters.params[f"model.layers.{l}.self_attn.o_proj.weight"].T

    
    n1 = rms_norm(x, rms_w1)
    
    n1 = grouped_query_attention(jnp.bfloat16(n1), dim, n_heads, n_kv_heads, context_length, q, k, v, o)
    x = x + n1
    
    n2 = rms_norm(x, rms_w2)
    n2 = feed_forward(jnp.bfloat16(n2), gate_proj, up_proj, down_proj)
    x = x + n2

    return x

def grouped_query_attention(x, dim, n_heads, n_kv_heads, context_length,
            q, k, v, o, 
            rope_base=10_000,
            rope_config=None,
            dtype=None):
    assert dim % n_heads == 0
    assert n_heads % n_kv_heads == 0
    n_repeat = n_heads // n_kv_heads
    head_dim = dim // n_heads
    n_seq = x.shape[0]
    assert x.shape[1] == dim

    cos, sin = PositionalEncodingHelper.precompute_rope_params(head_dim, rope_base, context_length, rope_config)
    xq = linear(x, q) # [n_seq, dim]
    xk = linear(x, k) # [n_seq, dim / n_repeat]
    xv = linear(x, v) # [n_seq, dim / n_repeat]

    xq = xq.reshape(n_seq, n_heads, head_dim)
    xk = xk.reshape(n_seq, n_kv_heads, head_dim)
    xv = xv.reshape(n_seq, n_kv_heads, head_dim)
    
    xq = xq.transpose(1, 0, 2) # [n_heads, n_seq, head_dim]
    xk = xk.transpose(1, 0, 2)
    xv = xv.transpose(1, 0, 2)

    
    xq = PositionalEncodingHelper.compute_rope(xq, cos, sin)
    xk = PositionalEncodingHelper.compute_rope(xk, cos, sin)
    
    
    if n_repeat > 1:        
        xk = jnp.repeat(xk, n_repeat, axis=0)
        xv = jnp.repeat(xv, n_repeat, axis=0)
    
    
    mask = None
    if n_seq > 1:
        mask = jnp.full((n_seq, n_seq), jnp.bfloat16(-1e10)) # -1e10 similar to -inf after softmax
        mask = jnp.triu(mask, k=1)

 
    
    attn_scores = xq @ xk.transpose(0, 2, 1) # [n_heads, n_seq, n_seq]
    
    if mask is not None:
        attn_scores = attn_scores + mask[None, :, :]

    attn_scores = softmax(attn_scores / xk.shape[-1]**0.5)

    context = linear(attn_scores, xv).transpose(1, 0, 2) # [n_seq, n_heads, head_dim]
    context = context.reshape(n_seq, dim)
    context = linear(context, o)
    return context

@jax.jit
def rms_norm(x, w, eps=1e-5):
    mean = (x ** 2).mean(-1, keepdims=True) + eps
    x = x / jnp.sqrt(mean)
    return x * w

@jax.jit
def silu(x):
    return x * (1 / (1 + jnp.exp(-x)))

@jax.jit
def softmax(x):
    exp_x = jnp.exp(x - jnp.max(x, axis = -1, keepdims = True))
    return exp_x / jnp.sum(exp_x, axis = -1, keepdims = True)

@jax.jit
def linear(x, w, b=None):
    if b is None:
        return x @ w
    return x @ w + b

@jax.jit
def feed_forward(x, gate_proj, up_proj, down_proj):
    x1 = linear(x, gate_proj)
    x2 = linear(x, up_proj)
    x = silu(x1) * x2
    x = linear(x, down_proj)
    return x
