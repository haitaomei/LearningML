import jax.numpy as jnp

from tqdm import tqdm
from tokenizer import Tokenizer, ChatFormat
from utils import Parameters
from llama3 import generate


def inference(prompt = "What does lion eat", n_tokens_to_generate = 40):
    tokenizer = Tokenizer("Llama-3.2-1B-Instruct/original/tokenizer.model")
    chat_tokenizer = ChatFormat(tokenizer)
    
    input_tokens = chat_tokenizer.encode(prompt)    
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = generate(input_tokens)
        next_token = jnp.argmax(logits[-1])
        input_tokens.append(int(next_token))

    output = input_tokens[len(input_tokens) - n_tokens_to_generate:]
    decode = chat_tokenizer.decode(output)
    print(decode)


###############################################################
# Note,  need to change the numbers in PositionalEncodingHelper
###############################################################
def main():
    Parameters.load_weight()

    inference(prompt="What does lion eat", n_tokens_to_generate=8)

if __name__ == "__main__":
    main()
