import torch
from torch import nn


class Generation(nn.Module):
    def generate(self, tokenizer, prompts, max_gen_len, temperature, top_p):
        prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        bsz = len(prompt_tokens)
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        total_len = max_gen_len + max_prompt_len
        tokens = torch.full((bsz, total_len), tokenizer.pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != tokenizer.pad_id

        for cur_pos in range(min_prompt_len, total_len):
            with torch.no_grad():
                logits = self(tokens[:, :cur_pos])
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == tokenizer.eos_id
            )
 
            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]

            # cut to eos tok if any
            if tokenizer.eos_id in toks:
                eos_idx = toks.index(tokenizer.eos_id)
                toks = toks[:eos_idx]

            out_tokens.append(toks)

        return [{"generation": tokenizer.decode(t)} for t in out_tokens]


    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
