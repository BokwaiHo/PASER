import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    """
    Prepare the model for k-bit training, including gradient checkpointing and input requiring gradients.
    """
    model = prepare_model_for_int8_training(model)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model

def get_peft_config(args):
    """
    Create a LoRA configuration based on the provided arguments.
    """
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

def get_peft_model(model, peft_config):
    """
    Apply LoRA to the model using the provided configuration.
    """
    return get_peft_model(model, peft_config)

def get_peft_model_state_dict(model, state_dict):
    """
    Extract the LoRA state dictionary from the full model state dictionary.
    """
    peft_state_dict = {}
    for k, v in state_dict.items():
        if "base_model.model" in k:
            peft_state_dict[k.replace("base_model.model.", "")] = v
    return peft_state_dict

def set_peft_model_state_dict(model, peft_model_state_dict):
    """
    Load the LoRA state dictionary into the model.
    """
    model.load_state_dict(peft_model_state_dict, strict=False)

def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def load_pruned_model(args):
    """
    Load the pruned model from a file.
    """
    pruned_dict = torch.load(args.prune_model, map_location='cpu')
    tokenizer, pruned_model = pruned_dict['tokenizer'], pruned_dict['model']
    return tokenizer, pruned_model

def save_model(model, output_dir):
    """
    Save the model to the specified output directory.
    """
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def load_model_for_inference(model_path, device):
    """
    Load a model for inference.
    """
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def move_model_to_device(model, device):
    """
    Move the model to the specified device (CPU or GPU).
    """
    return model.to(device)

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model_size(model):
    """
    Get the size of the model in megabytes.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb