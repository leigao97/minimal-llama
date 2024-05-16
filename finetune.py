import copy
import json
import logging

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama

import time

IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        torch.tensor(tokenizer.encode(text, bos=True, eos=True)) for text in strings
    ]
    input_ids = labels = [tokenized for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.ne(-1).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources, targets, tokenizer):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len-1] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path, tokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path, "r") as f:
            list_data_dict = json.load(f)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=-1
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(-1),
        )


def make_supervised_data_module(tokenizer, data_path):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    torch.manual_seed(1)

    tokenizer_path = "/home/lei/Project/llama2-7b/tokenizer.model"
    model_path = "/home/lei/Project/llama2-7b/consolidated.00.pth"
    data_path = "/home/lei/Project/llama/alpaca_data_200.json"

    # load model
    checkpoint = torch.load(model_path, map_location="cpu")
    model_args = ModelArgs()
    model_args.n_layers = 32  # for debugging purposes we only use 1 layer
    # torch.set_default_tensor_type(torch.cuda.HalfTensor) # for training we use fp32 weights
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=False)
    model.to("cuda")

    # load tokenizer
    tokenizer = Tokenizer(tokenizer_path)

    # create dataloader
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=data_path)
    dataloader = torch.utils.data.DataLoader(
        data_module["train_dataset"],
        batch_size=1,
        collate_fn=data_module["data_collator"],
        shuffle=True,
    )

    # Freeze model parameters other than lora weights
    for name, params in model.named_parameters():
        if "lora_" in name:
            params.requires_grad = True
        else:
            params.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_params:,d} || "
        f"trainable%: {100 * trainable_params / all_params:.2f}"
    )

    # prepare optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    scaler = torch.cuda.amp.GradScaler()

    iters_to_accumulate = 8

    start = time.time()
    for epoch in range(5):
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to("cuda")
            labels = batch['labels'].to("cuda")

            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(input_ids)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1, 32000)
                shift_labels = shift_labels.view(-1)

                loss = criterion(shift_logits, shift_labels)

                loss = loss / iters_to_accumulate
            
            scaler.scale(loss).backward()
            if (i + 1) % iters_to_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (i+1) % 50 == 0:
                print(loss.item())

    end = time.time()
    print(f"Time: {end - start}")
    # torch.save(model.state_dict(), "finetuned.pth")
    model_weights = model.state_dict()
    lora_weights = {k: v for k, v in model_weights.items() if "lora_" in k}
    torch.save(lora_weights, "lora_weights.pth")

if __name__ == "__main__":
    train()