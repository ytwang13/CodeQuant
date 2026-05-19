# CodeQuant: Unified Clustering and Quantization for Enhanced Outlier Smoothing in Low-Precision Mixture-of-Experts

<p align="center" style="font-size: 24px;">
    International Conference on Learning Representations (ICLR), 2026
</p>
<p align="center">
    <a href="https://openreview.net/forum?id=ATpchFiBQi">📄OpenReview</a>
</p>

This repository provides the official implementation of <strong>CodeQuant</strong>, a unified clustering and quantization framework for <strong>Mixture-of-Experts (MoE) Large Language Models (LLMs)</strong>, addressing activation outliers with fine-tuned rotation and robust clustering method, enabling efficient low-precision deployment.
![CodeQuant Overview](asset/codequant_overall.png)

## ⭐️Highlights
- Unified Rotation and Clustering framework for MoE LLMs low-precision deployment with carefully designed <strong>MoE-specific fine-tuning objectives</strong>.
- Fully offline quantization with <strong>no on-the-fly computation overhead</strong>. Achieving strong performance on language modeling, zero-shot QA tasks, and few-show mathematical reasoning.
- Lookup-table (LUT) based system for efficient deployment and inference, achieving <strong>4.15x speedup on CPU, and average 2.63x speedup on A100 GPU (simulator)</strong>.

## 🔧Requirements
Our implementation requires different [transformers](https://github.com/huggingface/transformers) versions for different models.
The DeepSeek model we used is [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) which requires lower version transformers. We used transformers==4.45.0. 
The best practice is to install the required packages separately. For DeepSeek-V2-Lite model, use the `requirements-deepseek.txt`. For other models (e.g. [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B), [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)), use the `requirements.txt`.
- DeepSeek-V2-Lite Model:
````shell
pip install -r requirements-deepseek.txt
````
- Other Models:
````shell
pip install -r requirements.txt
````

## 👨‍💻Pipeline
In our framework, we decouple the configuration and the pipeline. We have provided a set of examples for different models under `configs/` directory.
You can use our examples first to have a quick start of the pipeline. Then you can modify the configuration according to your needs.
### 🎯Run:
Our scrips are stored under `script/` directory. 
You can follow the following steps to reproduce our results.
- Step 1: run AOS to fine-tune the rotation matrix:
````shell
cd script/
python rotation_fine_tune_script.py --config model_name.yaml
````
- Step 2: run ACCF (set `permutation=True` and `weight_group_size` to some number to enable POG):
````shell
# cd script/
python cluster_fine_tune_script.py --config model_name.yaml
````
- Step 3: evaluate, we will use fake quantization for evaluation (for downstream tasks and math reasoning, we use the third-party evaluation tool [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)):
````shell
# cd script/
python evaluation_script.py --config model_name.yaml
````

### 🔍Config:
You can use our examples or create your own configurations. To create your own configuration, you can follow the example's structure and 
modify the following configuration parameters:
- accelerator:
  * `device`: The accelerator to use. If you use GPU, set it to `cuda`.
- path:
  * `rotation_data_path`: The path to save fine-tuned rotation matrix. We prefer the absolute path.
  * `cluster_data_path`: The path to save clustering results. We prefer the absolute path.
- model:
  * `model_name`: The huggingface model path. E.g. `Qwen/Qwen3-30B-A3B`.
- calibration:
  * `dataset_name`: The calibration dataset name. 
- common_setting:
  * `weight_group_size`: The group size for weight clustering. Set it to `-1` for embedding-wise setup. 
  * `input_group_size`: The group size for activation quantization. Set it to `-1` for embedding-wise setup.
  * `activation_quantization_format` (optional): Same formats as eval; used during rotation fine-tune (AOS) when set. R1 checkpoints are saved/loaded as `{model}_r1_{group|nongroup}_act_{format}.pt` (e.g. `_act_fp8e4m3`); omit the key to keep the legacy `{model}_r1_{group|nongroup}.pt` name.
- cluster:
  * `permutation`: The switch for POG. Set it to `True` for POG.
  * `max_sample`: The number of calibration samples to use for clustering fine-tune (ACCF).
  * `batch_size`: The batch size for clustering fine-tune (ACCF).
  * `max_length`: The maximum length of input tokens. Set it to a smaller value to save memory.
  * `epochs`: The number of epochs for clustering fine-tune (ACCF).
  * `fine_tune_lr`: The learning rate for clustering fine-tune (ACCF). Don't use scientific notation here (e.g. `1e-3`). Use decimal notation instead (e.g. `0.001`).
- rotation:
  * `max_sample`: The number of calibration samples to use for rotation fine-tune (AOS).
  * `batch_size`: The batch size for rotation fine-tune (AOS).
  * `max_length`: The maximum length of input tokens. Set it to a smaller value to save memory.
  * `epochs`: The number of epochs for rotation fine-tune (AOS).
  * `fine_tune_lr`: The learning rate for rotation fine-tune (AOS). Don't use scientific notation here (e.g. `1e-3`). Use decimal notation instead (e.g. `0.001`).
- eval:
  * `activation_quantization_bit`: The bitwidth for activation quantization (integer fake-quant).
  * `activation_quantization_format` (optional): FP8/NVFP4 activation fake-quant format. One of `fp8_e4m3`, `fp8_e5m2`, `e4m3`, `e5m2`, `nvfp4`, `nvfp4_plus`. When set, overrides integer activation quant during eval; weight quant is unchanged.
  * `weight_quantization_bit`: The bitwidth for weight quantization. This is only used for benchmark evaluation. If you evaluate a clustered model, this parameter will not be used.
  * `tasks`: The evaluation tasks. Use the format `task1,task2,...,taskN` where each task following naming convention of [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness).
  * `ppls`: The perplexity tasks. Use the format `ppl1,ppl2,...,pplN` where each task is a huggingface dataset path.

### Qwen3-4B FP8 / NVFP4 presets (`configs/qwen3_4_act_*`)

Based on `qwen3_4.yaml`. Each preset sets `activation_quantization_format` in `common_setting` and `eval`.

| Config | Format | Granularity (`input_group_size`) |
|--------|--------|----------------------------------|
| `qwen3_4_act_fp8_e4m3_{perchannel,perblock}` | fake FP8 E4M3 | `-1` / `128` |
| `qwen3_4_act_fp8_e5m2_{perchannel,perblock}` | fake FP8 E5M2 | `-1` / `128` |
| `qwen3_4_act_e4m3_{perchannel,perblock}` | native `float8_e4m3fn` cast | `-1` / `128` |
| `qwen3_4_act_e5m2_{perchannel,perblock}` | native `float8_e5m2` cast | `-1` / `128` |
| `qwen3_4_act_nvfp4_{perchannel,perblock}` | NVFP4 E2M1 | `-1` / `128` |
| `qwen3_4_act_nvfp4_plus_{perchannel,perblock}` | NVFP4 + FP8 scale | `-1` / `128` |

Example: `python evaluation_script.py --config qwen3_4_act_fp8_e4m3_perchannel.yaml` (from `script/`). R1 cache: `qwen_r1_nongroup_act_fp8e4m3.pt` (perchannel) or `qwen_r1_group_act_fp8e4m3.pt` (perblock).


## 📚Citation
If you find our work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{
    yin2026codequant,
    title={CodeQuant: Unified Clustering and Quantization for Enhanced Outlier Smoothing in Low-Precision Mixture-of-Experts},
    author={Xiangyang Yin and Xingyu Liu and Tianhua Xia and BO BAO and Vithursan Thangarasa and Valavan Manohararajah and Eric Sather and Sai Qian Zhang},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=ATpchFiBQi}
}
```

## 🤝Contributing
We thank the community for sharing their projects. Our project builds on top of the existing works:
- [QuaRot](https://github.com/spcl/QuaRot)
- [SpinQuant](https://github.com/facebookresearch/SpinQuant)
