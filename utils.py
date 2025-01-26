import json
import os
from typing import Union, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class Prompter:
    def __init__(self, template_name: str = "alpaca"):
        self.template_name = template_name
        self.template = self._load_template()

    def _load_template(self) -> dict:
        file_path = os.path.join("prompts", f"{self.template_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Template file {file_path} does not exist.")
        with open(file_path, "r") as f:
            return json.load(f)

    def generate_prompt(
        self,
        instruction: str,
        input: Union[str, None] = None,
        response: Union[str, None] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if response:
            res = f"{res}{response}"
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[-1].strip()

class ZeroPrompter:
    def generate_prompt(
        self,
        instruction: str,
        input: Union[str, None] = None,
        response: Union[str, None] = None,
    ) -> str:
        if input:
            prompt = f"{instruction}\n\n{input}"
        else:
            prompt = instruction
        if response:
            prompt = f"{prompt}\n\n{response}"
        return prompt

    def get_response(self, output: str) -> str:
        return output.strip()

class PASERDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized = tokenize(item['instruction'], self.tokenizer, self.args)
        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor(tokenized["labels"])
        }

def get_loaders(args, tokenizer):
    """
    Load and prepare datasets for training and evaluation.
    """
    # Load the dataset
    if args.data_path.startswith("data/"):
        # Load from local file
        with open(args.data_path, "r") as f:
            data = json.load(f)
    else:
        # Load from Hugging Face datasets
        data = load_dataset(args.data_path)

    if isinstance(data, dict):
        data = data["train"]

    # Split into train and validation sets
    train_val = data.train_test_split(test_size=args.val_set_size, shuffle=True, seed=42)
    train_data = train_val["train"]
    val_data = train_val["test"]

    # Create datasets
    train_dataset = PASERDataset(train_data, tokenizer, args)
    val_dataset = PASERDataset(val_data, tokenizer, args)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

def tokenize(prompt: str, tokenizer, args):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < args.cutoff_len
        and args.add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def load_tokenizer(args):
    """
    Load the tokenizer based on the base model.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return tokenizer

def generate_and_tokenize_prompt(data_point, prompter, tokenizer, args):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point.get("input", None),
        data_point.get("output", None),
    )
    tokenized_full_prompt = tokenize(full_prompt, tokenizer, args)
    if not args.train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point.get("input", None)
        )
        tokenized_user_prompt = tokenize(user_prompt, tokenizer, args)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if args.add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
    return tokenized_full_prompt