from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama
import torch

torch.manual_seed(1)

tokenizer_path = "/home/leig/Project/llama2-7b/tokenizer.model"
model_path = "/home/leig/Project/llama2-7b/consolidated.00.pth"

tokenizer = Tokenizer(tokenizer_path)

model_args = ModelArgs()
model = Llama(model_args)

checkpoint = torch.load(model_path, map_location="cpu")
model.load_state_dict(checkpoint, strict=False)
model.to("cuda")

prompts = [
    # For these prompts, the expected answer is the natural continuation of the prompt
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    """A brief message congratulating the team on the launch:

    Hi everyone,
    
    I just """,
    # Few shot prompt (providing a few examples before asking model to complete more);
    """Translate English to French:
    
    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese =>""",
]

model.eval()
with torch.no_grad(): # otherwise model will store all intermediate activations
    results = model.generate(tokenizer, prompts, max_gen_len=64, temperature=0.6, top_p=0.9)

for prompt, result in zip(prompts, results):
    print(prompt)
    print(f"> {result['generation']}")
    print("\n==================================\n")