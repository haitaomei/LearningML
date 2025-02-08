import jax
import jax.numpy as jnp

from gpt2 import generate, gelu, softmax, layer_norm, linear
from utils import load_encoder_hparams_and_params

def inference(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    print("input tokens:", len(input_ids))
    output_ids = generate([input_ids], params, hparams["n_head"], n_tokens_to_generate)
    output_txt = encoder.decode(output_ids[0])

    return output_txt

if __name__ == "__main__":
    input_text = "To create an array filled with any specified value"
    out = inference(input_text, 10)
    print(out)
    
    # import numpy as np
    # x = np.arange(24).reshape(1, 3, 8)

    # print("before split ", x.shape)
    # print(x)
    
    # x = jnp.array(np.split(x, 2, axis=-1))
    # print("after split ", x.shape)
    # print("first split is ", x[0])

    # x = jnp.concatenate(x, axis=-1)
    # print("after merge ", x.shape)
    # print(x)

    # import numpy as np
    # def myattention(q, kT, v, mask):
    #     return softmax(q @ kT / jnp.sqrt(q.shape[-1]) + mask) @ v
    # q = np.arange(24).reshape(3, 8)
    # k = q + 10
    # v = q + 20
    # mask = (1 - np.tri(q.shape[0], dtype=q.dtype)) * -1e10
    # print(myattention(q, k.T, v, mask))


    # q = np.arange(24).reshape(1, 3, 8)
    # k = q + 10
    # v = q + 20
    # print(myattention(q, k.transpose((0, 2, 1)), v, mask))
