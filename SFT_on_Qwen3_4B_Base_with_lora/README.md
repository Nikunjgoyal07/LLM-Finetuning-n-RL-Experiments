# SFT on Qwen3-4B-Base with LoRA (Reasoning)

Supervised fine-tune of **Qwen3-4B-Base** with **LoRA via Unsloth** to teach structured math reasoning:

```
<think> ... working out ... </think><SOLUTION> ... final answer ... </SOLUTION>
```

End-to-end Colab (Tesla T4) run: load base model → attach LoRA → format `unsloth/OpenMathReasoning-mini` into chat format → train with TRL `SFTTrainer` → test generation → save LoRA adapter.

## Folder Structure

```
SFT_on_Qwen3_4B_Base_with_lora/
├── SFT_on_Qwen3_4B_Base_with_lora.ipynb  # Full pipeline (16 cells, Colab + T4)
├── README.md                             # This file
└── qwen3-4b-reasoning-sft/               # Saved LoRA adapter (output of trainer.save_model)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json / tokenizer_config.json
    ├── vocab.json / merges.txt
    ├── added_tokens.json / special_tokens_map.json
    ├── chat_template.jinja
    ├── training_args.bin
    └── README.md                         # Auto-generated HF model card (mostly unfilled template)
```

> The nested `qwen3-4b-reasoning-sft/README.md` is the default PEFT/HF card. This parent README documents the actual experiment.

## Method

### 1. Base model + Unsloth setup

```python
from unsloth import FastLanguageModel

max_seq_length = 2048
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False,      # 16-bit LoRA, not QLoRA
    fast_inference = False,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7,
)
```

Runtime observed: `Tesla T4, 1 GPU, 14.563 GB, Torch 2.10.0+cu128, Unsloth 2026.9.11, Transformers 4.57.6, vLLM 0.18.0`.

### 2. LoRA config

```python
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

From `qwen3-4b-reasoning-sft/adapter_config.json`:

| Field | Value |
|---|---|
| `base_model_name_or_path` | `unsloth/Qwen3-4B-Base` |
| `peft_type` | `LORA`, `task_type: CAUSAL_LM` |
| `r` | `32` |
| `lora_alpha` | `64` |
| `lora_dropout` | `0.0` |
| `bias` | `none` |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| `peft_version` | `0.21.0` |
| Trainable | `66,060,288 / 4,088,528,384 (1.62%)` |

### 3. Reasoning format + chat template

Special markers:

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

Custom Jinja template (`qwen3-4b-reasoning-sft/chat_template.jinja`) prepends the system prompt + `eos_token`, concatenates `user` / `assistant` turns, and opens with `<think>` when `add_generation_prompt=True`:

```jinja
{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + eos_token }}{% set loop_messages = messages[1:] %}{% else %}{{ 'You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <think> and </think>.\nThen, provide your solution between <SOLUTION></SOLUTION>' + eos_token }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<think>' }}{% endif %}
```

Verified in-notebook:

```
user: "What is 1+1?"
assistant: "<think>I think it's 2.</think><SOLUTION>2</SOLUTION>"
user: "What is 2+2?"
→ template renders system + eos + history + trailing "<think>"
```

## Dataset

Source: `unsloth/OpenMathReasoning-mini`, split `cot` (19,252 rows, ~106 MB parquet).

Pipeline in notebook:

```python
from datasets import load_dataset
dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
dataset = dataset.to_pandas()[["expected_answer", "problem", "generated_solution"]]

# Keep only numeric answers
is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors="coerce").notnull()
dataset = dataset.iloc[np.where(is_number)[0]]  # -> 7507 rows
```

Formatting (`format_dataset`):

```python
thoughts = x["generated_solution"].replace("<think>","").replace("</think>","").strip()
final_prompt = "<think>" + thoughts + "</think>" + "<SOLUTION>" + expected_answer + "</SOLUTION>"
# -> [{"role":"system","content":system_prompt},
#     {"role":"user","content":problem},
#     {"role":"assistant","content":final_prompt}]
```

Length filter for T4 / 2048 context:

```python
dataset["N"] = dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
dataset = dataset.loc[dataset["N"] <= max_seq_length/2].copy()  # <=1024 tokens -> 63 rows
dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize=False)
dataset = Dataset.from_pandas(dataset)
```

| Stage | Rows |
|---|---|
| Raw `cot` split | 19,252 |
| Numeric `expected_answer` only | 7,507 |
| Tokenized length `<= 1024` | **63** |

Final HF `Dataset`: `features: [expected_answer, problem, generated_solution, Messages, N, text], num_rows: 63`.

> This aggressive `19252 → 63` reduction is the biggest caveat — a demo-scale SFT imposed by `max_seq_length=2048` on a T4.

## Training Setup

Install (via `uv pip` in Colab):

```
torch==2.12.1 unsloth==2026.9.11 transformers==5.5.0 datasets==4.3.0 accelerate bitsandbytes
vllm==0.18.0   # downgrades torch->2.10.0, transformers->4.57.6 in the logged run
```

`trl.SFTTrainer`:

```python
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)
trainer.train()
```

Effective: `63 examples × 2 epochs, batch 4 → 32 steps`. Unsloth enables padding-free training + gradient offload + double buffering.

## Results

Training loss log:

| Step | Loss |
|---|---|
| 5 | 0.3580 |
| 10 | 0.2986 |
| 15 | 0.2978 |
| 20 | 0.2379 |
| 25 | 0.2144 |
| 30 | 0.1866 |

Final: `global_step=32, train_loss=0.260, runtime=182.5s, 0.69 samples/s, 0.175 steps/s`.

### Qualitative check

Prompt (first train row, messages `[:2]` + `add_generation_prompt=True`):

> Jenifer has 82 cents in pennies and nickels. Her younger brother mistook all her nickels for dimes and counted the total as $1.47. How many pennies does Jenifer have?

Model generates full CoT (`P + 5N = 82`, `P + 10N = 147` → `N=13` → `P=17`) and answers `17`, matching `expected_answer='17'`.

```python
text = tokenizer.apply_chat_template(dataset[0]["Messages"][:2], tokenize=False, add_generation_prompt=True)
_ = model.generate(**tokenizer(text, return_tensors="pt").to("cuda"),
                   temperature=0, max_new_tokens=1024, streamer=TextStreamer(tokenizer))
```

## How to Reproduce

1. Open `SFT_on_Qwen3_4B_Base_with_lora.ipynb` in Colab with GPU (tested T4).
2. Run install cells (Unsloth + vLLM).
3. Run model + LoRA cells — adjust `max_seq_length`, `gpu_memory_utilization` if OOM.
4. Run chat-template + dataset cells — expects `unsloth/OpenMathReasoning-mini`.
5. Run `SFTTrainer` + `trainer.train()` (~3 min on T4 for 63 rows).
6. Run inference cell to sanity-check reasoning output.
7. Run `trainer.save_model("qwen3-4b-reasoning-sft")` + zip cell to regenerate the adapter folder.

## Load the Saved Adapter for Inference

```python
from unsloth import FastLanguageModel
from transformers import TextStreamer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen3-4b-reasoning-sft",  # local LoRA dir in this folder
    max_seq_length=2048,
    load_in_4bit=False,
)

messages = [
    {"role": "system", "content": open("qwen3-4b-reasoning-sft/chat_template.jinja").read()},  # or reuse system_prompt from notebook
    {"role": "user", "content": "What is 1+1?"},
]
# Preferred: tokenizer already carries chat_template.jinja
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
_ = model.generate(**tokenizer(text, return_tensors="pt").to("cuda"),
                   temperature=0, max_new_tokens=1024,
                   streamer=TextStreamer(tokenizer, skip_prompt=False))
```

Note: `adapter_config.json` has `inference_mode: true`; for further training set it to `False` or re-attach via `FastLanguageModel.get_peft_model`.

## Artifacts

- `adapter_model.safetensors` — LoRA weights only (base weights not included; need `Qwen3-4B-Base`).
- `adapter_config.json` — LoRA hyperparams (see table above).
- `chat_template.jinja`, `tokenizer.*`, `vocab.json`, `merges.txt` — Qwen3 tokenizer + reasoning template (`<think>` supported, `model_max_length: 32768`, `padding_side: left`).
- `training_args.bin` — saved `SFTConfig`.
- `qwen3-4b-reasoning-sft.zip` (created in Colab, not committed) — zipped adapter for upload/share.

## Limitations & Next Steps

- **Tiny train set (63 rows):** length filter `<= max_seq/2` discards ~99% of numeric data. Increase `max_seq_length` (e.g. 4096+) on larger GPU or use packing/truncation to keep more data.
- **2 epochs, no eval split:** loss decreases but no held-out math accuracy reported. Add eval + exact-match on `<SOLUTION>` extraction.
- **16-bit LoRA:** `load_in_4bit=False` is VRAM-heavy on T4; QLoRA (`True`) would allow longer context / bigger batch.
- **Generic model card:** nested `README.md` is unfilled HF template — fill in data, procedure, and intended use before pushing to Hub.
- **Single qualitative test:** only one generation shown; test multi-problem reasoning and check for extraneous-solution / tag-leak issues.
