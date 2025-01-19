import jax.numpy as jnp

# hparams:
# {
#   "n_vocab": 50257, # number of tokens in our vocabulary
#   "n_ctx": 1024, # maximum possible sequence length of the input
#   "n_embd": 768, # embedding dimension (determines the "width" of the network)
#   "n_head": 12, # number of attention heads (n_embd must be divisible by n_head)
#   "n_layer": 12 # number of layers (determines the "depth" of the network)
# }

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    pass

def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = jnp.argmax(logits[-1])
        inputs.append(int(next_id))

    return inputs[len(inputs) - n_tokens_to_generate:]

def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt( 2 / jnp.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = jnp.exp(x - jnp.max(x, axis = -1, keepdims = True))
    return exp_x / jnp.sum(exp_x, axis = -1, keepdims = True)

def layer_norm(x, g, b, eps: float = 1e-5):
    """
    Layer normalization ensures that the inputs for each layer are always 
    within a consistent range, which is supposed to speed up and stabilize 
    the training process. Like Batch Normalization, the normalized output 
    is then scaled and offset with two learnable vectors gamma and beta. 
    The small epsilon term in the denominator is used to avoid a division 
    by zero error.
    """
    mean = jnp.mean(x, axis = -1, keepdims = True)
    variance = jnp.var(x, axis = -1, keepdims = True)
    x = (x - mean) / jnp.sqrt(variance + eps)
    return g * x + b

def linear(x, w, b):
    return x @ w + b
