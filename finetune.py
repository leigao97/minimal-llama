import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset

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


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = utils.jload(data_path)
        with open(data_path, "r") as f:
            list_data_dict = json.load(f)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    # We will remove HF model
    model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

    # We will remove HF tokenizer
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )

    # For HF llama tokenizer, by default:
    # tokenizer.pad_token = None, tokenizer.pad_token_id = None
    # tokenizer.eos_token = </s>, tokenizer.eos_token_id = 2
    # tokenizer.bos_token = <s>, tokenizer.bos_token_id = 1
    # tokenizer.unk_token = <unk>, tokenizer.unk_token_id = 0
    # We set pad_token_id to unk_token_id, instead of adding extra special token and resize embedding table.
    # If pad_token_id = None, it will cause error during batch-tokenization.
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    data_path = "/home/leig/Project/llama/alpaca_data_dummy.json"
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=data_path)

    # For debugging purpose
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]

    print(train_dataset[0])
    print(train_dataset[0]['input_ids'].shape, train_dataset[0]['labels'].shape)
    print(tokenizer.decode(train_dataset[0]['input_ids']))

    # freeze most of the model param to save memory for debugging purpose
    for name, param in model.named_parameters():
        if "lm_head" not in name:
            param.requires_grad = False

    # We will remove HF training arguments and trainer
    training_args = transformers.TrainingArguments(output_dir="/home/leig/Project/llama/output", overwrite_output_dir=True, num_train_epochs=1)
    trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()

if __name__ == "__main__":
    train()