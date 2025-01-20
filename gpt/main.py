import jax
import jax.numpy as jnp

from gpt2 import generate, gelu, softmax, layer_norm, linear
from utils import load_encoder_hparams_and_params

def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    # encoder is the BPE tokenizer
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    
    # ids = encoder.encode("How are you doing today.")
    # print([encoder.decoder[i] for i in ids])
    # # The size of the vocabulary
    # print(len(encoder.decoder))

    input_ids = encoder.encode(prompt)
    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_txt = encoder.decode(output_ids)

    return output_txt

if __name__ == "__main__":
    # a = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], jnp.int32)
    # print(a, "\n", a[jnp.arange(len(a)), [0, 2, 1]])
    # print(gelu(jnp.array([[1, 2], [-2, 0.5]])))
    # x = softmax(jnp.array([[2, 100], [-5, 0]]))
    # print(x, "\n", x.sum(axis = -1))
    # x = jnp.array([[2, 2, 3], [-5, 0, 1]])
    # x = layer_norm(x, g=jnp.ones(x.shape[-1]), b=jnp.zeros(x.shape[-1]))
    # print(x, "\n variance=", x.var(axis=-1),"\tmean=", x.mean(axis=-1))
    # causal_mask = (1 - jnp.tri(4, dtype=jnp.float32)) * -1e10
    # print(softmax(causal_mask))
    # pass

    out = main("Alan Turing theorized that computers would one day become", 8)
    print(out)
