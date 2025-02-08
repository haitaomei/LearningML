import jax
import jax.numpy as jnp

from gpt2 import generate, gelu, softmax, layer_norm, linear
from utils import load_encoder_hparams_and_params

def inference(prompts, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    """
    Make sure each prompt in prompts with the same length
    """
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)    
    inputs = []
    for prompt in prompts:
        tokens = encoder.encode(prompt)
        # print(len(tokens))
        assert len(tokens) + n_tokens_to_generate < hparams["n_ctx"]
        inputs.append(tokens)
    
    output_toks = generate(inputs, params, hparams["n_head"], n_tokens_to_generate)
    outputs = []
    for output_tok in output_toks:
        outputs.append(encoder.decode(output_tok))

    return outputs

if __name__ == "__main__":
    input_texts = ["Google LLC is an American multinational corporation and technology company", 
                   "Labrado is a dog breed that originated in",]
    out = inference(input_texts, 13)
    print(out)

    # import numpy as np
    # a = np.array([[1, 2], [3, 4]])
    # b = np.array([[5, 6]])
    # print(a.shape, b.T.shape)
    # print(np.concatenate((a, b.T), axis=1))
    
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
