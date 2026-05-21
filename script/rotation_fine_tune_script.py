import argparse
import os
import random
import numpy as np
import yaml
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Union
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.rotation_utils import load_or_create_R1, create_optimizer
from utils.dataset_utils import CalibrationDataset
from utils.quantization_utils import activation_quantizer, r1_checkpoint_filename
from utils.rotation_utils import fuse_weight
from utils.model_utils import get_model


def collect_activation_hook(module_name: str,
                            mode: str,
                            activation_dict: dict):
    """Collect detached activations only — no grad graph on the frozen model."""

    def hook(module: nn.Module,
             input_args: Tuple[torch.Tensor, ...],
             output_tensor_or_tuple: Union[torch.Tensor, Tuple]) -> None:
        if mode == "output":
            if isinstance(output_tensor_or_tuple, torch.Tensor):
                act = output_tensor_or_tuple.detach()
            elif isinstance(output_tensor_or_tuple, tuple):
                if len(output_tensor_or_tuple) > 0 and isinstance(output_tensor_or_tuple[0], torch.Tensor):
                    act = output_tensor_or_tuple[0].detach()
                else:
                    return
            else:
                return
            activation_dict[module_name] = act
        elif mode == "input":
            if len(input_args) > 0 and isinstance(input_args[0], torch.Tensor):
                act = input_args[0].detach()
            elif isinstance(input_args, torch.Tensor):
                act = input_args.detach()
            else:
                return
            activation_dict[module_name] = act
    return hook


def rotation_fine_tune(model: nn.Module,
                       model_type: str,
                       tokenizer: nn.Module,
                       processor: nn.Module,
                       input_group_size: int,
                       activation_quantization_bit: int,
                       dataset_name: str,
                       calibration_samples: int,
                       batch_size: int,
                       max_length: int,
                       epochs: int,
                       lr: float,
                       device: str,
                       activation_format: Optional[str] = None):
    R1 = load_or_create_R1(mode="online",
                           device=device,
                           dim=model.config.hidden_size)
    R1_optimizer = create_optimizer(R1, lr=lr)

    output_name_pattern = r".*(input_layernorm|post_attention_layernorm).*"
    output_regex = re.compile(output_name_pattern)

    dataset = CalibrationDataset(dataset_name, calibration_samples)
    g = torch.Generator()
    g.manual_seed(42)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)

    model.eval()
    for _ in tqdm(range(epochs), leave=False):
        for batch in tqdm(dataloader, leave=False):
            # Step 1: collect activations under no_grad (no full-model graph retained)
            activation_dict = {}
            handles = {}
            for name, module in model.named_modules():
                if output_regex.search(name):
                    hook = collect_activation_hook(name, "output", activation_dict)
                    handles[name] = module.register_forward_hook(hook)

            if model_type == "qwen":
                processed = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    for text in batch
                ]
                inputs = tokenizer(
                    processed, return_tensors="pt",
                    padding=True, truncation=True, max_length=max_length
                ).to(device)
            elif model_type == "mixtral":
                inputs = tokenizer(
                    batch, return_tensors="pt",
                    padding=True, truncation=True, max_length=max_length
                ).to(device)
            else:
                processed = [
                    processor.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    for text in batch
                ]
                inputs = tokenizer(
                    processed, return_tensors="pt",
                    padding=True, truncation=True, max_length=max_length
                ).to(device)

            with torch.no_grad():
                model(**inputs)

            for h in handles.values():
                h.remove()
            handles.clear()
            del inputs

            # Step 2: backprop R1 one layernorm at a time (bounded peak memory)
            R1_optimizer.zero_grad()
            for name, act in activation_dict.items():
                rot_output = act.float() @ R1.weight.to(act.device)
                quant_rot_output = activation_quantizer(
                    (rot_output,),
                    input_group_size,
                    activation_quantization_bit,
                    activation_format=activation_format,
                )
                layer_loss = F.mse_loss(rot_output, quant_rot_output)
                layer_loss.backward()
                del act, rot_output, quant_rot_output, layer_loss

            activation_dict.clear()
            torch.cuda.empty_cache()

            tqdm.write(f"[DEBUG] rotation matrix gradient: {torch.norm(R1.parametrizations.weight.original.grad)}")
            R1_optimizer.step()
    return R1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input parser")
    parser.add_argument('--config', type=str, required=True, help='config file name')
    args = parser.parse_args()

    with open(f"../configs/{args.config}", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config is not None, "[ERROR] config is None."

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_name = config["model"]["model_name"]
    device = config["accelerator"]["device"]

    model, model_type, tokenizer, processor, model_config = get_model(model_name)

    # fuse norm weight into linear weight
    fuse_weight(model, model_type)
    print(f"[INFO] model {model_name} fused.")

    # params
    common_params = config["common_setting"]
    dataset_name = config["calibration"]["dataset_name"]
    rotate_params = config["rotation"]

    # path
    save_path = config["path"]["rotation_data_path"]
    os.makedirs(save_path, exist_ok=True)

    # run
    activation_quantization_format = common_params.get("activation_quantization_format")
    if activation_quantization_format:
        print(f'[INFO] activation quantization format: {activation_quantization_format}')

    r1 = rotation_fine_tune(model=model,
                            model_type=model_type,
                            tokenizer=tokenizer,
                            processor=processor,
                            input_group_size=common_params["input_group_size"],
                            activation_quantization_bit=common_params["activation_quantization_bit"],
                            activation_format=activation_quantization_format,
                            dataset_name=dataset_name,
                            calibration_samples=rotate_params["max_sample"],
                            batch_size=rotate_params["batch_size"],
                            max_length=rotate_params["max_length"],
                            epochs=rotate_params["epochs"],
                            lr=rotate_params["fine_tune_lr"],
                            device=device)

    r1_ckpt_name = r1_checkpoint_filename(
        model_type,
        common_params["input_group_size"],
        activation_quantization_format,
    )
    r1_ckpt = {
        "r1": r1.cpu().state_dict(),
        "dim": model.config.hidden_size,
    }
    torch.save(r1_ckpt, os.path.join(save_path, r1_ckpt_name))
    print(f"[INFO] R1 saved to {os.path.join(save_path, r1_ckpt_name)}")
