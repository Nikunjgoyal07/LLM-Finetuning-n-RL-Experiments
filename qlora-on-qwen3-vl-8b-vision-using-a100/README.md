# QLoRA on Qwen3-VL-8B Vision (LaTeX OCR) on A100

Vision fine-tune of **Qwen3-VL-8B-Instruct** with **QLoRA via Unsloth** for image-to-LaTeX OCR:

```
math-crop image + "Write the LaTeX representation for this image." -> LaTeX string
```

End-to-end run in two notebooks: train on **Modal A100-SXM4-40GB** (`qlora-on-qwen3-vl-8b-vision-using-a100.ipynb`) → inference + qualitative eval on **Colab T4** (`Qwen3_VL_LaTeX_OCR_—_Inference_&_Evaluation.ipynb`) → save LoRA adapter to `qwen_qlora/`.

> Training status (honest): `trainer.train()` was **interrupted at [31/135, Epoch 0.22/1] after ~21:07** (`KeyboardInterrupt`). The adapter in `qwen_qlora/` is therefore a **partial-epoch checkpoint**, not a full 1-epoch run. Loss had already converged sharply before the interrupt (see Results).

## Folder Structure

```
qlora-on-qwen3-vl-8b-vision-using-a100/
├── qlora-on-qwen3-vl-8b-vision-using-a100.ipynb       # Training pipeline (24 cells, Modal + A100)
├── Qwen3_VL_LaTeX_OCR_—_Inference_&_Evaluation.ipynb  # Inference + qualitative eval (18 cells, Colab + T4)
├── README.md                                          # This file
└── qwen_qlora/                                        # Saved LoRA adapter (output of model.save_pretrained)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json / tokenizer_config.json
    ├── vocab.json / merges.txt
    ├── added_tokens.json / special_tokens_map.json
    ├── chat_template.jinja
    ├── preprocessor_config.json / video_preprocessor_config.json
    └── README.md                                      # Auto-generated HF model card (mostly unfilled template)
```

> The nested `qwen_qlora/README.md` is the default PEFT/HF card. This parent README documents the actual experiment.

## Method

### 1. Base model + Unsloth Vision setup

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    load_in_4bit = True,  # 4-bit QLoRA
    use_gradient_checkpointing = "unsloth",
)
```

Runtime observed (training nb): `NVIDIA A100-SXM4-40GB (39.494 GB), Torch 2.12.1+cu130, CUDA 13.0, Unsloth 2026.9.12, Transformers 4.57.1, TRL 0.22.2`. Pre-train reserve `7.699 GB`, `~8.2 GB` in `nvidia-smi`. Modal volume used for output: `/__modal/volumes/vo-ELo9w1eKOpX9sD5Qf3nkF6`.

Inference nb runtime: `Tesla T4 15 GB, Torch 2.11.0+cu128, Triton 3.6.0, Unsloth 2026.9.12, Transformers 4.57.1, Bfloat16=FALSE`.

### 2. LoRA config (QLoRA, vision + language)

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
```

From `qwen_qlora/adapter_config.json`:

| Field | Value |
|---|---|
| `base_model_name_or_path` | `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` |
| `peft_type` | `LORA`, `task_type: CAUSAL_LM` |
| `r` | `16` |
| `lora_alpha` | `16` |
| `lora_dropout` | `0` |
| `bias` | `none` |
| `target_modules` | regex covering vision (`qkv/proj/linear_fc1/fc2`) + LLM (`q/k/v/o/gate/up/down_proj`) — see `adapter_config.json:target_modules` |
| `peft_version` | `0.21.1` |
| Trainable | `51,346,944 / 8,818,470,640 (0.58%)` |

### 3. Prompt format

Single instruction for all samples:

```python
instruction = "Write the LaTeX representation for this image."
```

Training conversion (`convert_to_conversation`):

```python
def convert_to_conversation(sample):
    return { "messages": [
        { "role": "user", "content": [
            {"type": "text",  "text": instruction},
            {"type": "image", "image": sample["image"]} ]},
        { "role": "assistant", "content": [
            {"type": "text",  "text": sample["text"]} ]},
    ]}
converted_dataset = [convert_to_conversation(s) for s in dataset]  # 68,686 msgs
```

Inference prompt (chat template + generation prompt):

```python
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": instruction}]}]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
```

## Dataset

Source: `unsloth/LaTeX_OCR`, `features: ['image', 'text']`.

| Split | Rows | Used for |
|---|---|---|
| `train` | `68,686` | SFT train (all rows, converted to messages) |
| `test` | `7,632` | Downloaded but **not used** in either notebook |

Sample (`dataset[0]`):

- `image`: `PIL PNG RGB 160x40`
- `text`: `{ \frac { N } { M } } \in { \bf Z } , { \frac { M } { P } } \in { \bf Z } , { \frac { P } { Q } } \in { \bf Z }`

Other samples probed in inference nb: `dataset[1]` (`120x50`), `dataset[15]`.

## Training Setup

Install (training nb, Modal):

```
pip install unsloth
uv pip install transformers==4.57.1
uv pip install --no-deps trl==0.22.2
```

Install (inference nb, Colab T4):

```
pip install unsloth sentencepiece protobuf datasets huggingface_hub hf_transfer
pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton
pip install transformers==4.57.1
pip install latex2sympy2  # installed but never used
```

`trl.SFTTrainer` with Unsloth vision collator:

```python
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),  # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 128,
        gradient_accumulation_steps = 4,  # effective batch 512
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)
trainer_stats = trainer.train()
```

Planned: `68,686 examples x 1 epoch, eff batch 512 -> 135 steps`. Unsloth enables gradient offload + double buffering.

Pre-train sanity check in training nb (`FastVisionModel.for_inference`, `max_new_tokens=128, temperature=1.5, min_p=0.1`, `TextStreamer`): base model already renders `dataset[0]` correctly before any training.

Save (training nb):

```python
model.save_pretrained("qwen_qlora")
tokenizer.save_pretrained("qwen_qlora")
shutil.make_archive("qwen_qlora", "zip", "qwen_qlora")  # local
shutil.make_archive("/__modal/volumes/vo-ELo9w1eKOpX9sD5Qf3nkF6/qwen_qlora", "zip", "qwen_qlora")  # Modal Volume
```

## Inference & Evaluation Notebook

Loads the partial adapter as a fused unit (Unsloth auto-fetches base):

```python
adapter_path = "/content/qwen_qlora"  # unzipped from /content/qwen_qlora.zip (185M)
model, tokenizer = FastVisionModel.from_pretrained(adapter_path, load_in_4bit=True)
FastVisionModel.for_inference(model)  # 2x faster inference
```

Greedy decode + strip prompt:

```python
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, use_cache=True)
latex = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
display(Math(latex)); display(image)
```

No training in this notebook. No metric loop — only `ACTUAL vs GENERATED` print + `Math()` render on 2–3 samples.

## Results

### Training log (interrupted)

```
[ 31/135 21:07 < 1:15:45, 0.02 it/s, Epoch 0.22/1 ] -> KeyboardInterrupt
```

| Step | Loss |
|---|---|
| 1 | 1.3321 |
| 2 | 1.3409 |
| 3 | 1.3163 |
| 4 | 1.2382 |
| 5 | 1.0684 |
| 6 | 0.8785 |
| 7 | 0.6210 |
| 8 | 0.4405 |
| 9 | 0.3284 |
| 10 | 0.2376 |
| 12 | 0.1694 |
| 15 | 0.1096 |
| 16 | 0.0825 |
| 18 | 0.0594 |
| 25 | 0.0506 |
| 28 | 0.0451 |
| 29 | 0.0520 |

Rapid drop `1.33 -> ~0.05` by step ~25, then plateau. No eval loss, no CER/WER/BLEU.

### Qualitative checks

| Sample | Actual | Generated (with adapter) | Verdict |
|---|---|---|---|
| `dataset[0]` train nb post-save (`temperature=1.5`) | `{ \frac { N } { M } } \in { \bf Z } , ...` | `\frac { N } { M } \in { \bf Z } , ...<\|im_end\|>` | Correct (spacing/style only) |
| `dataset[1]` infer nb (`do_sample=False`) | `D _ { \mu } ^ { \alpha \beta } \bar { A } _ { \mu } ^ { \alpha \beta } = 0 ,` | identical | Perfect |
| `dataset[15]` infer nb (`do_sample=False`) | `{ \ln } J _ { \theta } = ... a _ { 5 / 2 } ...` | `\ln J _ { \theta } = ... \alpha _ { 5 / 2 } ...` | Near miss: missing `{ }` around `\ln`, `a` vs `\alpha` |

## How to Reproduce

Training (needs ~40GB GPU, logged A100):

1. Open `qlora-on-qwen3-vl-8b-vision-using-a100.ipynb` on Modal (or Colab with A100).
2. Run install cells (`unsloth + transformers==4.57.1 + trl==0.22.2`).
3. Run `FastVisionModel.from_pretrained` + `get_peft_model` cells.
4. Run dataset cells — expects `unsloth/LaTeX_OCR` (`train` 68,686 rows).
5. Run pre-train inference cell (optional sanity check).
6. Run `SFTTrainer` + `trainer.train()` (planned 135 steps; logged run stopped at 31).
7. Run `model.save_pretrained("qwen_qlora")` + zip cells to regenerate the adapter.

Inference (tested T4):

1. Open `Qwen3_VL_LaTeX_OCR_—_Inference_&_Evaluation.ipynb` in Colab with T4.
2. Upload `qwen_qlora.zip` (185M) to `/content`, unzip to `/content/qwen_qlora`.
3. Run install + `FastVisionModel.from_pretrained(adapter_path)` cells.
4. Run generation cells for `idx=1`, `idx=15` (adjust `idx` for more samples).

Load the saved adapter for inference:

```python
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(
    "qwen_qlora",  # local LoRA dir in this folder
    load_in_4bit = True,
)
FastVisionModel.for_inference(model)
# then apply_chat_template + tokenizer(image, text) + model.generate(do_sample=False) as above
```

Note: `adapter_config.json` has `inference_mode: true`; for further training set it to `False` or re-attach via `FastVisionModel.get_peft_model`.

## Artifacts

- `adapter_model.safetensors` — LoRA weights only (base weights not included; need `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`).
- `adapter_config.json` — LoRA hyperparams (see table above).
- `tokenizer.json / tokenizer_config.json / vocab.json / merges.txt / added_tokens.json / special_tokens_map.json / chat_template.jinja` — Qwen3-VL tokenizer + chat template.
- `preprocessor_config.json / video_preprocessor_config.json` — vision preprocessor (image + video) configs.
- `qwen_qlora.zip` (created in notebook, 185M in Colab run) — zipped adapter for upload/share.
- `outputs/` (trainer dir, not committed) — checkpoints/logs if run completes.

## Limitations & Next Steps

- **Partial training (31/135 steps, 0.22 epoch):** adapter is an early checkpoint. Run full epoch(s) to completion.
- **No quantitative eval:** no CER/WER/BLEU/exact-match, no held-out test (`test` 7,632 rows unused). Add metric loop over `test` split; `latex2sympy2` is already installed for symbolic equivalence.
- **Train-split reuse for eval:** inference probes (`idx=1, 15`) come from `train`, so scores are optimistic. Eval on `test` split.
- **Slow step time:** `0.02 it/s` on A100 with `batch 128x4` + 2048 length — profile `UnslothVisionDataCollator` / image resize (Unsloth defaults to 512, no default image size in model).
- **Sampling mismatch:** train nb uses `temperature=1.5, min_p=0.1`, inference nb uses greedy `do_sample=False`. Fix one decoding protocol for fair comparison.
- **Generic model card:** nested `qwen_qlora/README.md` is unfilled HF template — fill in data, procedure, and intended use before pushing to Hub.
- **Single-GPU memory:** `7.7–8.2 GB` reserved on A100 leaves headroom; larger `r`/batch or full-finetune of vision tower is feasible.
