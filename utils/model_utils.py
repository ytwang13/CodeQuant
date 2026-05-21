import torch
import torch.nn as nn

from typing import Dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from accelerate import dispatch_model

model_prefix_dict = {
    "microsoft/Phi-mini-MoE-instruct": "phi",
    "Qwen/Qwen3-30B-A3B": "qwen",
    "Qwen/Qwen3-4B": "qwen",
    "mistralai/Mixtral-8x7B-v0.1": "mixtral",
    "deepseek-ai/DeepSeek-V2-Lite": "deepseek",
}

# HF dense Qwen3 checkpoints may leave lm_head randomly initialized unless tied to embeddings.
_DENSE_QWEN_UNTIED_LM_HEAD = frozenset({
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-0.6B",
})


def is_qwen_moe(model: nn.Module) -> bool:
    """True for Qwen MoE (layer.mlp.experts); False for dense Qwen3 MLP."""
    return hasattr(model.model.layers[0].mlp, "experts")


def is_qwen_fused_moe_experts(experts: nn.Module) -> bool:
    """HF Qwen3 MoE: ``Qwen3MoeExperts`` with stacked ``gate_up_proj`` / ``down_proj``."""
    return hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")


def qwen_mlp_experts(mlp: nn.Module) -> nn.Module | None:
    """Return ``mlp.experts`` when present."""
    return getattr(mlp, "experts", None)


def make_gpu_map(num_layers: int, num_devices: int):
    dm = {"model.embed_tokens": 0}
    layers_per_device = num_layers // num_devices
    for i in range(num_layers):
        dev = min(i // layers_per_device, num_devices - 1)
        dm[f"model.layers.{i}"] = dev
    dm["model.norm"] = num_devices - 1
    dm["lm_head"] = num_devices - 1
    return dm


def split_gpu(model, num_devices: int):
    device_map = make_gpu_map(len(model.model.layers), num_devices)
    print(device_map)
    return dispatch_model(model, device_map=device_map)


@torch.no_grad()
def _tie_lm_head_from_embeddings(model: nn.Module) -> None:
    """Copy embedding weights into lm_head (matches code2_dev dense Qwen3 load)."""
    embed_tokens = model.model.embed_tokens
    lm_head = model.lm_head
    lm_head_device = lm_head.weight.device
    lm_head.weight.data = embed_tokens.weight.data.clone().to(
        dtype=lm_head.weight.dtype, device=lm_head_device
    )
    if lm_head.bias is not None and getattr(embed_tokens, "bias", None) is not None:
        lm_head.bias.data = embed_tokens.bias.data.to(
            dtype=lm_head.weight.dtype, device=lm_head_device
        )


def get_model(model_name: str,
              force_device: dict = None):
    model_type = model_prefix_dict[model_name]

    device_map = "auto" if force_device is None else force_device

    if model_name in _DENSE_QWEN_UNTIED_LM_HEAD:
        model_config = AutoConfig.from_pretrained(
            model_name,
            attn_implementation="eager",
            trust_remote_code=True,
            tie_word_embeddings=False,
        )
        print(f"[INFO] model config {model_name} loaded (dense Qwen3, untied embeddings).")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
            config=model_config,
        )
        _tie_lm_head_from_embeddings(model)
        print(f"[INFO] tied lm_head weights to embed_tokens for {model_name}.")
    else:
        model_config = AutoConfig.from_pretrained(
            model_name, attn_implementation="eager", trust_remote_code=True
        )
        print(f"[INFO] model config {model_name} loaded.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
            config=model_config,
        )

    # model = split_gpu(model, 2) #optional
    print(model.device)
    model.config.attn_implementation = "eager"
    model.eval()
    print(f"[INFO] model {model_name} loaded.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if model_type == "mixtral":
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[INFO] tokenizer {model_name} loaded.")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print(f"[INFO] processor {model_name} loaded.")

    return model, model_type, tokenizer, processor, model_config


@torch.no_grad()
def replace_model_weight(model: nn.Module,
                         weight_dict: Dict[str, torch.Tensor]) -> nn.Module:
    for name, module in model.named_modules():
        if name in weight_dict:
            print(f"[INFO] replace {name} with quantized weight")
            module.weight.data = weight_dict[name].to(module.weight.device)

    return model

@torch.no_grad()
def replace_model_weight_rebuttal(
    model: nn.Module,
    weight_dict: Dict[str, torch.Tensor],
    # target_indices,
    verbose: bool = True
) -> nn.Module:
    target_indices = [24,25,26]
    target_lists = []
    for index in target_indices:
        target_lists.append(f"model.layers.{index}.")

    for name, module in model.named_modules():
        if name in weight_dict:
            flag = 0
            for item in target_lists:
                if item in name:
                    flag = 1
                    break
            
            if flag == 0:
                continue
                
            print(f"[INFO] replace {name} with quantized weight")
            module.weight.data = weight_dict[name].to(module.weight.device)

    return model