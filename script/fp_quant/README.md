# FP activation quantization scripts (Qwen3-4B)

Pipeline mirrors [`../qwen3-4b.sh`](../qwen3-4b.sh) but uses configs under `configs/act_fp_quant/`. Cache dirs and R1 filenames are resolved from `activation_quantization_format` (see repo README).

## Quick start

```bash
source /home/wyt/miniconda3/bin/activate code
cd /mnt/hdd/wyt/code1

# One preset, full pipeline (default: fp8_e4m3_perchannel)
bash script/fp_quant/qwen3-4b-fp.sh all

# Preset shorthand + step
bash script/fp_quant/qwen3-4b-fp.sh e4m3_perblock rotation
bash script/fp_quant/qwen3-4b-fp.sh nvfp4_plus_perblock cluster
bash script/fp_quant/qwen3-4b-fp.sh fp8_e5m2_perchannel eval

# Explicit config path
CONFIG=act_fp_quant/qwen3_4_act_e4m3_perblock.yaml bash script/fp_quant/qwen3-4b-fp.sh all
```

## Presets

| Shorthand | Format | Granularity |
|-----------|--------|-------------|
| `fp8_e4m3_perchannel` / `_perblock` | fake FP8 E4M3 | perchannel / perblock |
| `fp8_e5m2_perchannel` / `_perblock` | fake FP8 E5M2 | perchannel / perblock |
| `e4m3_perchannel` / `_perblock` | native `float8_e4m3fn` | perchannel / perblock |
| `e5m2_perchannel` / `_perblock` | native `float8_e5m2` | perchannel / perblock |
| `nvfp4_perchannel` / `_perblock` | NVFP4 E2M1 | perchannel / perblock |
| `nvfp4_plus_perchannel` / `_perblock` | NVFP4 + FP8 scale | perchannel / perblock |

## Batch runs

```bash
# All 12 presets sequentially (one log per preset: log_fp_<preset>.log)
bash script/fp_quant/run-all.sh
bash script/fp_quant/run-all.sh eval

# All presets in background
bash script/fp_quant/nohup-all.sh
STEP=rotation bash script/fp_quant/nohup-all.sh
```

## Steps

Same as `qwen3-4b.sh`: `1` / `rotation` → `2` / `cluster` → `3` / `eval` → `all`.
