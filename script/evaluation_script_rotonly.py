import yaml
import argparse
import os
import random
import numpy as np
import torch

from utils.evaluation_utils import evaluation
from utils.quantization_utils import (
    group_postfix,
    r1_checkpoint_filename,
    resolve_fp_data_path,
)
from utils.model_utils import get_model, replace_model_weight
from utils.rotation_utils import fuse_rotation, fuse_weight, load_or_create_R1
from utils.permutation_utils import permutation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input parser")
    parser.add_argument('--config', type=str, required=True, help='config file name')
    parser.add_argument(
        '--rotation-lr',
        type=float,
        default=None,
        help='rotation fine_tune_lr used for R1 checkpoint lookup (default: config rotation.fine_tune_lr)',
    )
    args = parser.parse_args()

    with open(f"../configs/{args.config}", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config is not None, "[ERROR] config is None."

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load model
    model_name = config["model"]["model_name"]
    device = config["accelerator"]["device"]

    # evaluate
    model, model_type, tokenizer, processor, model_config = get_model(model_name)

    # eval tasks
    tasks = config["eval"]["tasks"]
    print(f'[INFO] tasks: {tasks}')
    ppls = config["eval"]["ppls"]
    print(f'[INFO] ppls: {ppls}')

    # eval config
    activation_quantization_bit = config["eval"]["activation_quantization_bit"]
    print(f'[INFO] activation quantization bit: {activation_quantization_bit}')
    activation_quantization_format = config["eval"].get("activation_quantization_format")
    weight_quantization_bit = config["eval"]["weight_quantization_bit"]
    print(f'[INFO] weight quantization bit: {weight_quantization_bit}')
    activation_group_size = config["common_setting"]["input_group_size"]
    print(f'[INFO] activation group size: {activation_group_size}')
    weight_group_size = config["common_setting"]["weight_group_size"]
    print(f'[INFO] weight group size: {weight_group_size}')

    # fuse RMSNorm weight
    fuse_weight(model, model_type)
    print(f"[INFO] model {model_name} RMSNorm weight fused.")

      # load rotation R1 (suffix reflects AOS activation quant in common_setting)
    input_group_size = config["common_setting"]["input_group_size"]
    r1_act_format = config["common_setting"].get("activation_quantization_format")
    rotation_save_path = resolve_fp_data_path(
        config["path"]["rotation_data_path"],
        "rotation",
        r1_act_format,
        input_group_size,
    )
    cluster_save_path = resolve_fp_data_path(
        config["path"]["cluster_data_path"],
        "cluster",
        r1_act_format,
        input_group_size,
    )
    rotation_lr = (
        args.rotation_lr
        if args.rotation_lr is not None
        else config["rotation"]["fine_tune_lr"]
    )
    print(f"[INFO] rotation fine_tune_lr: {rotation_lr}")
    R1_save_dir = os.path.join(
        rotation_save_path,
        r1_checkpoint_filename(
            model_type,
            input_group_size,
            r1_act_format,
            fine_tune_lr=rotation_lr,
        ),
    )
    print(f"[INFO] rotation cache dir: {rotation_save_path}")
    print(f"[INFO] cluster cache dir: {cluster_save_path}")
    print(f"[INFO] loading R1 checkpoint: {R1_save_dir}")
    postfix = group_postfix(input_group_size)
    R1 = load_or_create_R1(mode="offline", device=device, save_dir=R1_save_dir)
    R1 = R1.weight.detach()

    # cache R1 on each GPU
    gpu_count = torch.cuda.device_count()
    R1_per_gpu = {}
    if gpu_count > 1:
        print(f"[INFO] Detected {gpu_count} GPUs, creating R1 copies...")
        for i in range(gpu_count):
            R1_per_gpu[f"cuda:{i}"] = R1.to(f"cuda:{i}")
            print(f"[INFO] R1 copied to cuda:{i}")
    else:
        R1_per_gpu["cuda:0"] = R1.to(device=device)

    # fuse rotation matrix
    fuse_rotation(model, model_type, R1_per_gpu, None)
    print(f"[INFO] model {model_name} rotation matrix fused.")
    del R1, R1_per_gpu

    ppl_results, task_results, clean_results = evaluation(model=model,
                                                          model_type=model_type,
                                                          tokenizer=tokenizer,
                                                          tasks=tasks,
                                                          ppls=ppls,
                                                          quantization_bit=activation_quantization_bit,
                                                          input_group_size=activation_group_size,
                                                          activation_format=activation_quantization_format,
                                                          is_baseline=False)
    print(str(activation_quantization_bit))
    print(ppl_results)
    print(clean_results)
