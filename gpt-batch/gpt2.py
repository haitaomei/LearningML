import jax
import jax.numpy as jnp

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    max_seq_length = len(inputs[0])
    for i in range(len(inputs)):
        max_seq_length = max(max_seq_length, len(inputs[i]))

    x = wte[inputs] + wpe[range(max_seq_length)]

    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)

    x = layer_norm(x, **ln_f)

    return x @ wte.T

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head = n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

@jax.jit
def ffn(x, c_fc,  c_proj):
    a = gelu(linear(x, **c_fc))
    x = linear(a, **c_proj)
    return x

@jax.jit
def attention(q, kT, v, mask):
    return softmax(q @ kT / jnp.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head): # [batch, n_seq, n_embd] -> [batch, n_seq, n_embd]
    x = linear(x, **c_attn) # [batch, n_seq, n_embd] -> [batch, n_seq, 3*n_embd]
    
    qkv = jnp.split(x, 3, axis=-1) # [batch, n_seq, 3*n_embd] -> [batch, 3, n_seq, n_embd]

    q = qkv[0] # [batch, n_seq, n_embd]
    k = qkv[1] # [batch, n_seq, n_embd]
    v = qkv[2] # [batch, n_seq, n_embd]

    n_seq = q.shape[1]
    n_embd = q.shape[2]
    assert n_embd % n_head == 0

    # # the following code is equivalent to the commented code
    # q = jnp.array(jnp.split(q, n_head, axis=-1)) # [n_head, batch, n_seq, n_embd//n_head]
    # k = jnp.array(jnp.split(k, n_head, axis=-1))
    # v = jnp.array(jnp.split(v, n_head, axis=-1))

    # kT = k.transpose((0, 1, 3, 2)) # [n_head, batch, n_embd//n_head, n_seq]
    # mask = (1 - jnp.tri(n_seq, dtype=x.dtype)) * -1e10

    # attn = attention(q, kT, v, mask) # [n_head, batch, n_seq, n_embd//n_head]
    # x = jnp.concatenate(attn, axis=-1) # [batch, n_seq, n_embd]
    # x = linear(x, **c_proj)
    # return x

    batch = q.shape[0]
    n_seq = q.shape[1]
    n_embd = q.shape[2]
    assert n_embd % n_head == 0
    q = q.reshape(batch, n_seq, n_head, n_embd//n_head) # [batch, n_seq, n_head, n_embd//n_head]
    k = k.reshape(batch, n_seq, n_head, n_embd//n_head)
    v = v.reshape(batch, n_seq, n_head, n_embd//n_head)
    q = q.transpose((2, 0, 1, 3)) # [n_head, batch, n_seq, n_embd//n_head]
    k = k.transpose((2, 0, 1, 3))
    v = v.transpose((2, 0, 1, 3))

    kT = k.transpose((0, 1, 3, 2)) # [n_head, batch, n_embd//n_head, n_seq]
    mask = (1 - jnp.tri(n_seq, dtype=x.dtype)) * -1e10

    attn = attention(q, kT, v, mask) # [n_head, batch, n_seq, n_embd//n_head]
    attn = attn.transpose((1, 2, 0, 3)) # [batch, n_seq, n_head, n_embd//n_head]
    x = attn.reshape(batch, n_seq, n_embd) # [batch, n_seq, n_embd]

    x = linear(x, **c_proj)
    return x

@jax.jit
def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt( 2 / jnp.pi) * (x + 0.044715 * x**3)))

@jax.jit
def softmax(x):
    exp_x = jnp.exp(x - jnp.max(x, axis = -1, keepdims = True))
    return exp_x / jnp.sum(exp_x, axis = -1, keepdims = True)

@jax.jit
def layer_norm(x, g, b, eps: float = 1e-5):
    mean = jnp.mean(x, axis = -1, keepdims = True)
    variance = jnp.var(x, axis = -1, keepdims = True)
    x = (x - mean) / jnp.sqrt(variance + eps)
    return g * x + b

@jax.jit
def linear(x, w, b):
    return x @ w + b


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    inputs = jnp.array(inputs)
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_tokens = jnp.argmax(logits, axis=-1)[:,-1] # [batch]
        next_tokens = jnp.expand_dims(next_tokens, axis=0) # [1, batch]
        next_tokens = next_tokens.T # [batch, 1]
        inputs = jnp.concatenate([inputs, next_tokens], axis=-1)
    
    return inputs.tolist()
