import os
import argparse
import torch
import transformers
from datasets import load_dataset

from paser import PASER
from utils import Prompter, ZeroPrompter, get_loaders, load_tokenizer
from model_utils import prepare_model_for_kbit_training, get_peft_config, get_peft_model_state_dict, set_peft_model_state_dict

def parse_arguments():
    parser = argparse.ArgumentParser(description='PASER: Post-training Data Selection for Efficient Pruned Large Language Model Recovery')

    # Model and data arguments
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-hf", help='Base model name or path')
    parser.add_argument('--prune_model', type=str, required=True, help='Path to pruned model')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='Data path or dataset name')
    parser.add_argument('--output_dir', type=str, default="./output", help='Output directory')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='Micro batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate') # Alpaca: 1e-4, LaMini: 5e-5
    parser.add_argument('--cutoff_len', type=int, default=256, help='Max sequence length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='Validation set size')

    # LoRA configuration
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA r dimension')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, 
                        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
                        help='Comma-separated list of target modules for LoRA')

    # PASER arguments
    parser.add_argument('--max_selected_data', type=int, default=20000, help='Maximum number of data points to select')
    parser.add_argument('--num_clusters', type=int, default=10, help='Number of clusters for semantic-structural clustering')
    parser.add_argument('--loss_ratio_threshold', type=float, default=0.1, help='Threshold for selecting data based on capability loss')

    # Misc
    parser.add_argument('--wandb_project', type=str, default="", help='Weights & Biases project name')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set up distributed training if applicable
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")

    # Set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, config=args)

    # Load models and tokenizer
    print("Loading models and tokenizer...")
    original_model = transformers.AutoModelForCausalLM.from_pretrained(args.base_model)
    pruned_dict = torch.load(args.prune_model, map_location='cpu')
    tokenizer, pruned_model = pruned_dict['tokenizer'], pruned_dict['model']

    original_model.to(device)
    pruned_model.to(device)

    # Prepare data
    print("Preparing data...")
    train_loader, val_loader = get_loaders(args, tokenizer)

    # Initialize PASER
    print("Initializing PASER...")
    paser = PASER(args)
    selected_data = paser.select_data(pruned_model, original_model, train_loader.dataset.data, tokenizer)

    # Prepare model for training
    print("Preparing model for training...")
    pruned_model = prepare_model_for_kbit_training(pruned_model)
    peft_config = get_peft_config(args)
    pruned_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.prune_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Set up trainer
    print("Setting up trainer...")
    trainer = transformers.Trainer(
        model=pruned_model,
        train_dataset=selected_data,
        eval_dataset=val_loader.dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=100,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="wandb" if args.wandb_project else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )

    # Train model
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save the final model
    print("Saving final model...")
    pruned_model.save_pretrained(os.path.join(args.output_dir, "final_model"))

    print("Training complete!")

if __name__ == "__main__":
    main()