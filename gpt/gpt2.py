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
    """
    token + position embedding
    
    wte is a [n_vocab, n_embd] matrix
    It acts as a lookup table, where the ith row in the matrix corresponds 
    to the learned vector for the ith token in our vocabulary.
    wte[inputs]   [n_seq] -> [n_seq, n_embd]
    
    One quirk of the original transformer architecture is that it doesn't take 
    into account position. Now we support position embedding
    wpe is a [n_ctx, n_embd] matrix
    The ith row of the matrix contains a vector that encodes information 
    about the ith position in the input.
    wpe[range(len(inputs))]   [n_seq] -> [n_seq, n_embd]
    
    Choosing a larger n_embd value allows us to control how wide our network is 
    (for example, GPT-3 uses an embedding size of 12288)
    """
    x = wte[inputs] + wpe[range(len(inputs))]

    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head) # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab, also called language modeling head
    x = layer_norm(x, **ln_f) # [n_seq, n_embd] -> [n_seq, n_embd]

    # reusing the embedding matrix, we can use another separate learned matrix.
    # using the same has advantages:
    # Space efficiency
    # Since it is both responsible for mapping both to words and from words, 
    # in theory, it may learn a richer representation compared to the 2 different ones
    return x @ wte.T # [n_seq, n_embd] -> [n_seq, n_vocab]

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head = n_head)
    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

def ffn(x, c_fc,  c_proj):
    # project up
    a = gelu(linear(x, **c_fc)) # [n_seq, n_embd] -> [n_seq, 4*n_embd]
    # project back down
    x = linear(a, **c_proj) # [n_seq, 4*n_embd] -> [n_seq, n_embd]
    return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
    def attention_without_masking(q, k, v):
        return softmax(q @ k.T / jnp.sqrt(q.shape[-1])) @ v
    
    Masking makes sure that no word can see future during training
    """
    return softmax(q @ k.T / jnp.sqrt(q.shape[-1]) + mask) @ v


def self_attention(x, c_attn, c_proj): # [n_seq, n_embd] -> [n_seq, n_embd]
    """
    When q, k, and v all come from the same source, we are performing self-attention 
    (i.e. letting our input sequence attend to itself)

    def self_attention_simple_version_without_linear(x, w_fc, w_proj):
        # The following is equivalent to
        # q = x @ w_q # [n_seq, n_embd] @ [n_embd, n_embd] -> [n_seq, n_embd]
        # k = x @ w_k # [n_seq, n_embd] @ [n_embd, n_embd] -> [n_seq, n_embd]
        # v = x @ w_v # [n_seq, n_embd] @ [n_embd, n_embd] -> [n_seq, n_embd]
        x = x @ w_fc # [n_seq, n_embd] @ [n_embd, 3*n_embd] -> [n_seq, 3*n_embd]
        q, k, v = jnp.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

        # perform self attention
        x = attention(q, k, v) # [n_seq, n_embd] -> [n_seq, n_embd]

        # out projection
        x = x @ w_proj # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

        return x
    
    We also add causal by creating the mask metric for softmax
    0 -1e10 -1e10 -1e10 -1e10
    0   0   -1e10 -1e10 -1e10
    0   0     0   -1e10 -1e10
    0   0     0     0   -1e10
    0   0     0     0     0 
    """
    # qkv projections, and split
    x = linear(x, **c_attn) # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    q, k, v = jnp.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

    # perform self attention
    causal_mask = (1 - jnp.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]    
    x = attention(q, k, v, causal_mask) # [n_seq, n_embd] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj) # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

    return x

def mha(x, c_attn, c_proj, n_head): # [n_seq, n_embd] -> [n_seq, n_embd]
    """
    We can further improve our implementation by performing n_head separate 
    attention computations, splitting our queries, keys, and values into heads
    """
    x = linear(x, **c_attn) # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    qkv = jnp.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(map(lambda x: jnp.split(x, n_head, axis=-1), qkv)) # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]
    
    causal_mask = (1 - jnp.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]
    
    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]
    
    # merge heads
    x = jnp.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x


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
