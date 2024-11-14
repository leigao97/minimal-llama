import argparse
from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama
import torch


def inference(args):
    torch.manual_seed(1)

    tokenizer = Tokenizer(args.tokenizer_path)

    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model_args = ModelArgs()
    model = Llama(model_args)
    model.half() # Run inference in FP16
    model.load_state_dict(checkpoint, strict=False)

    model.to("cuda")

    prompts = [
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]

    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    model.eval()
    results = model.generate(tokenizer, prompt_tokens, max_gen_len=128, temperature=0.7, top_p=0.9)

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")

    args = parser.parse_args()
    inference(args)
