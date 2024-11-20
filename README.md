# Minimal Implementation for Llama2 Inference and LoRA Fine-Tuning

This repository provides a simple and minimal implementation for performing inference and Low-Rank Adaptation (LoRA) fine-tuning on Llama2-7B models (need 40GB GPU memory). It is designed with minimal dependencies (only `torch` and `sentencepiece`) to provide a straightforward setup.

### Download Model and Tokenizer
* [HuggingFace Llama2 Model Weights and Tokenizer](https://huggingface.co/meta-llama/Llama-2-7b/tree/main)
* [HuggingFace Model Downloading Tutorial](https://huggingface.co/docs/hub/en/models-downloading)

### Install Required Dependencies
```
pip install torch sentencepiece
```

### Run Inference
```
python inference.py --tokenizer_path /path_to/tokenizer.model --model_path /path_to/consolidated.00.pth
```

### Run LoRA Fine-tuning
We use [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) with only 200 samples for quick experimentation. LoRA implmenetation is under the `llama` folder.
```
python finetune.py --tokenizer_path /path_to/tokenizer.model --model_path /path_to/consolidated.00.pth --data_path alpaca_data_200_samples.json
```


### Reference
* [meta-llama](https://github.com/meta-llama/llama)
* [stanford-alpaca](https://github.com/tatsu-lab/stanford_alpaca)
* [microsoft-lora](https://github.com/microsoft/LoRA)
