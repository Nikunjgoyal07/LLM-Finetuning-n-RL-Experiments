# GRPO on Qwen3-4B-Base with LoRA (Reasoning)

RL fine-tune of **Qwen/Qwen3-4B-Base** with **LoRA via Unsloth + TRL GRPO + vLLM** to teach structured math reasoning:

```
<think> ... working out ... </think><SOLUTION> ... final answer ... </SOLUTION>
```

End-to-end run on **L40S 44GB**: load base → attach LoRA (r=32) → short SFT warmup on `unsloth/OpenMathReasoning-mini` → GRPO on `open-r1/DAPO-Math-17k-Processed` with 4 rule-based rewards → save LoRA to `grpo_saved_lora` → qualitative checks via `model.fast_generate`.

> Folder-name note: despite `...-on-gsm8k-dataset` in the name, the notebook does **not** load GSM8K. It uses `unsloth/OpenMathReasoning-mini` (SFT warmup) + `open-r1/DAPO-Math-17k-Processed` (GRPO).

## Folder Structure

```
grpo-on-qwen3-4b-base-with-lora-on-gsm8k-dataset/
├── grpo-on-qwen3-4b-base-with-lora-on-gsm8k-dataset.ipynb  # Full pipeline (44 cells)
└── README.md                                              # This file
```

No adapter weights committed (`*.safetensors` ignored, see root `.gitignore`). Regenerate `grpo_saved_lora/` / `outputs/` by re-running the notebook.

## Method

### 1. Base model + LoRA (Unsloth)

```python
from unsloth import FastLanguageModel

max_seq_length = 2048
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False,      # 16-bit LoRA
    fast_inference = True,     # vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
    lora_alpha = 64,  # = rank * 2
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

Observed runtime: `NVIDIA L40S 44.39GB, Torch 2.10.0+cu128, Unsloth 2026.9.11, Transformers 4.57.6, vLLM 0.18.0`. Trainable: `66,060,288 / 4,088,528,384 (1.62%)`.

### 2. Reasoning format + chat template

```python
reasoning_start = "<think>"
reasoning_end   = "</think>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"
system_prompt = """You are given a problem.
Think about the problem and provide your working out.
Place it between <think> and </think>.
Then, provide your solution between <SOLUTION></SOLUTION>"""
```

Custom Jinja template prepends `system_prompt + eos_token`, concatenates user/assistant turns, and opens with `<think>` when `add_generation_prompt=True`.

### 3. Phase A — SFT warmup (same 63-row recipe as SFT experiment)

```python
dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
# keep numeric expected_answer only -> 7507 rows
# keep tokenized length <= max_seq_length/2 (1024) -> 63 rows
```

`trl.SFTTrainer` with `per_device_train_batch_size=8, num_train_epochs=4, lr=2e-4, warmup=5, optim=adamw_8bit`:
`global_step=32, train_loss=0.4577, runtime~48s`.

Sanity check passes: pennies/nickels problem → `P+5N=82, P+10N=147 → N=13, P=17`, matches `expected_answer='17'`.

### 4. Phase B — GRPO

Dataset:

```python
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
# 14116 rows -> prompt=[system,user], answer=solution
# token-length filter at 90th percentile (193 tokens) -> 12728 rows kept
```

Rewards (`trl.GRPOTrainer(reward_funcs=[...])`):

| Reward | Logic |
|---|---|
| `match_format_exactly` | `+3.0` if regex `<think>...</think><SOLUTION>answer</SOLUTION>(eos?)` matches at end |
| `match_format_approximately` | `+0.5` each for single occurrence of `</think>`, `<SOLUTION>`, `</SOLUTION>` else `-1.0` |
| `check_answer` | `+5.0` exact string, `+3.5` stripped, `+2.0/+1.5` if `float(guess)/float(true)` in `[0.9,1.1]/[0.8,1.2]`, else `-2.5` (numeric miss) / `-4.5` (non-numeric), `-2.0` if no extract |
| `check_numbers` | first number in `<SOLUTION>` vs truth as float (comma-tolerant): `+3.5` / `-1.5`, `-2.5` if none, logs Q/A/response every 5 steps |

Config:

```python
GRPOConfig(
    temperature=1.0, learning_rate=5e-6, weight_decay=0.01,
    warmup_ratio=0.1, lr_scheduler_type="linear", optim="adamw_8bit",
    per_device_train_batch_size=4, gradient_accumulation_steps=4,  # eff batch 16
    num_generations=4, max_prompt_length=194, max_completion_length=1854,
    max_steps=100, save_steps=100, output_dir="outputs", report_to="none",
    vllm_sampling_params=SamplingParams(min_p=0.1, top_p=1.0, top_k=-1, seed=3407),
)
```

`12728 examples × 100 steps, eff batch 16, 4 generations/step`.

Save + verify:

```python
model.save_lora("grpo_saved_lora")
# assert no all-zero A/B matrices in adapter_model.safetensors
```

## Results

No held-out accuracy curve; evidence is training logs + post-hoc `fast_generate` with loaded LoRA:

| Prompt | Output | Verdict |
|---|---|---|
| `sqrt(144)?` | CoT + `\boxed{12}` + `<SOLUTION>12</SOLUTION>` | Correct |
| `20% off 500?` (×2 runs) | `500×0.20=100 → 400`, also `500×0.80=400` + `<SOLUTION>400</SOLUTION>` | Correct, stable |
| `x^2=49?` | `x=7 or x=-7` + `<SOLUTION>7 and -7</SOLUTION>` | Correct |
| `LCM(6,8)?` | prime-factor CoT → `24` + `<SOLUTION>24</SOLUTION>` | Correct |
| `3^5 vs 5^3?` | `243 vs 125` (+ log check) → `<SOLUTION>3^5</SOLUTION>` | Correct |
| `sqrt(101)?` (pre-LoRA `fast_generate`, no `lora_request`) | unrelated subset-counting trace ending `1920` | Wrong-task leakage — not from final adapter |

Training-trace observations:
- Format rewards bias toward trailing `<SOLUTION>` extraction, which is what `check_answer`/`check_numbers` score.
- Numeric tolerance (`0.9–1.1 → +2.0`) gives partial credit for close floats.
- Minor artifacts remain: stray tokens before `<SOLUTION>` (`azienk`, `webElementX`, `hieronta`), duplicated `\boxed{}` + `<SOLUTION>`, verbose CoT.

Bottom line: after SFT warmup + 100 GRPO steps, the saved LoRA reliably emits `<think>` reasoning + machine-parseable `<SOLUTION>` and solves the 5 elementary probes above.

## How to Reproduce

1. Open `grpo-on-qwen3-4b-base-with-lora-on-gsm8k-dataset.ipynb` on a ≥40GB GPU (logged L40S).
2. Run `uv pip install` cells (Unsloth + `vllm==0.18.0`) + `apt install libcurand-dev-12-8`.
3. Run model + LoRA + chat-template cells.
4. Run SFT warmup cells (~48s) + generation sanity check.
5. Run DAPO-Math mapping + reward-fn + length-filter + `GRPOTrainer(...).train()` (100 steps).
6. Run `model.save_lora("grpo_saved_lora")` + `fast_generate(..., lora_request=model.load_lora("grpo_saved_lora"))` probes.

Inference with saved adapter:

```python
from vllm import SamplingParams
sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=4096)
out = model.fast_generate(
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text
```

## Limitations & Next Steps

- **Name/dataset mismatch:** rename folder or switch to actual GSM8K if GSM8K comparison is intended.
- **Tiny SFT warmup (63 rows):** `N <= 1024` filter discards ~99% of numeric OpenMathReasoning; use larger `max_seq_length`/packing on L40S.
- **Short RL (100 steps, no eval split):** no GSM8K/MATH held-out exact-match; add eval + `<SOLUTION>` extractor scoring.
- **Reward hacking surface:** ratio-based partial credit + format bonuses can reward well-formed wrong answers; ablate weights.
- **Single-GPU vLLM + LoRA:** `temperature=1.0, num_generations=4` tuned for L40S memory; lower on smaller GPUs.
- **Leakage/verbosity:** stray suffix tokens and over-long CoT; add KL penalty tuning, stop-string handling, and output-length penalty.
