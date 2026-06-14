import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def write_notebook(filename, cells):
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS / filename
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "L4",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


SETUP_INSTALL = r"""
# Run this once at the top of a fresh Colab runtime.
%pip -q install -U transformers datasets accelerate peft trl bitsandbytes pillow matplotlib pandas
"""


DRIVE_SETUP = r"""
from pathlib import Path

try:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_DIR = Path("/content/drive/MyDrive/smolvlm_lora_lab")
except Exception:
    PROJECT_DIR = Path.cwd() / "smolvlm_lora_lab"

DATA_DIR = PROJECT_DIR / "synthetic_glyph_vqa"
IMAGE_DIR = DATA_DIR / "images"
ADAPTER_DIR = PROJECT_DIR / "smolvlm_256m_glyph_lora_adapter"

for path in [PROJECT_DIR, DATA_DIR, IMAGE_DIR, ADAPTER_DIR]:
    path.mkdir(parents=True, exist_ok=True)

print("Project dir:", PROJECT_DIR)
print("Dataset dir:", DATA_DIR)
print("Adapter dir:", ADAPTER_DIR)
"""


SYNTHETIC_DATASET = r"""
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

RNG_SEED = 42
NUM_SCENES = 48

COLORS = {
    "teal": (20, 184, 166),
    "magenta": (217, 70, 239),
    "amber": (245, 158, 11),
    "lime": (132, 204, 22),
    "blue": (59, 130, 246),
    "rose": (244, 63, 94),
}

SHAPES = ["circle", "square", "triangle", "diamond"]
COUNTS = [1, 2, 3, 4]
DIRECTIONS = ["up", "right", "down", "left"]

COLOR_CODE = {
    "teal": "luma",
    "magenta": "voro",
    "amber": "kivo",
    "lime": "nexo",
    "blue": "safi",
    "rose": "pavo",
}

SHAPE_CODE = {
    "circle": "orbit",
    "square": "block",
    "triangle": "spire",
    "diamond": "facet",
}

ACTION_CODE = {
    "up": "mip-forward",
    "right": "mip-right",
    "down": "mip-back",
    "left": "mip-left",
}


def draw_arrow(draw, direction, box):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    if direction == "up":
        draw.line((cx, y1, cx, y0 + 10), fill=(20, 20, 20), width=7)
        draw.polygon([(cx, y0), (cx - 13, y0 + 18), (cx + 13, y0 + 18)], fill=(20, 20, 20))
    elif direction == "down":
        draw.line((cx, y0, cx, y1 - 10), fill=(20, 20, 20), width=7)
        draw.polygon([(cx, y1), (cx - 13, y1 - 18), (cx + 13, y1 - 18)], fill=(20, 20, 20))
    elif direction == "right":
        draw.line((x0, cy, x1 - 10, cy), fill=(20, 20, 20), width=7)
        draw.polygon([(x1, cy), (x1 - 18, cy - 13), (x1 - 18, cy + 13)], fill=(20, 20, 20))
    else:
        draw.line((x1, cy, x0 + 10, cy), fill=(20, 20, 20), width=7)
        draw.polygon([(x0, cy), (x0 + 18, cy - 13), (x0 + 18, cy + 13)], fill=(20, 20, 20))


def make_scene_image(scene, size=384):
    image = Image.new("RGB", (size, size), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, size - 18, size - 18), radius=24, outline=(35, 35, 35), width=4)
    draw.rectangle((34, 34, size - 34, size - 34), outline=(214, 214, 206), width=2)

    color = COLORS[scene["color"]]
    cx, cy = size // 2, size // 2 + 8
    r = 76
    if scene["shape"] == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=(20, 20, 20), width=4)
    elif scene["shape"] == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=color, outline=(20, 20, 20), width=4)
    elif scene["shape"] == "triangle":
        draw.polygon([(cx, cy - 92), (cx - 92, cy + 72), (cx + 92, cy + 72)], fill=color, outline=(20, 20, 20))
        draw.line([(cx, cy - 92), (cx - 92, cy + 72), (cx + 92, cy + 72), (cx, cy - 92)], fill=(20, 20, 20), width=4)
    else:
        draw.polygon([(cx, cy - 96), (cx - 96, cy), (cx, cy + 96), (cx + 96, cy)], fill=color, outline=(20, 20, 20))
        draw.line([(cx, cy - 96), (cx - 96, cy), (cx, cy + 96), (cx + 96, cy), (cx, cy - 96)], fill=(20, 20, 20), width=4)

    dot_y = size - 72
    start_x = cx - ((scene["count"] - 1) * 28) // 2
    for i in range(scene["count"]):
        x = start_x + i * 28
        draw.ellipse((x - 9, dot_y - 9, x + 9, dot_y + 9), fill=(20, 20, 20))

    draw_arrow(draw, scene["direction"], (size - 96, 46, size - 42, 100))
    return image


def glyph_answer(scene):
    return f"{COLOR_CODE[scene['color']]}-{SHAPE_CODE[scene['shape']]}-{scene['count']}"


QUESTION_SPECS = [
    (
        "glyph_code",
        "What is the synthetic glyph code for this panel? Answer only the code.",
        glyph_answer,
    ),
    (
        "dot_count",
        "How many black marker dots are shown? Answer with one number.",
        lambda scene: str(scene["count"]),
    ),
    (
        "arrow_direction",
        "What direction does the black arrow point? Answer one word.",
        lambda scene: scene["direction"],
    ),
    (
        "control_token",
        "What control token belongs to the arrow direction? Answer only the token.",
        lambda scene: ACTION_CODE[scene["direction"]],
    ),
]


def create_dataset(force=False):
    metadata_path = DATA_DIR / "metadata.jsonl"
    if metadata_path.exists() and not force:
        rows = [json.loads(line) for line in metadata_path.read_text().splitlines()]
        return pd.DataFrame(rows)

    random.seed(RNG_SEED)
    combos = [
        {"color": c, "shape": s, "count": n, "direction": d}
        for c in COLORS
        for s in SHAPES
        for n in COUNTS
        for d in DIRECTIONS
    ]
    random.shuffle(combos)
    scenes = combos[:NUM_SCENES]

    rows = []
    for scene_id, scene in enumerate(scenes):
        image = make_scene_image(scene)
        file_name = f"scene_{scene_id:03d}.png"
        image.save(IMAGE_DIR / file_name)

        split = "train" if scene_id < int(NUM_SCENES * 0.8) else "eval"
        for question_kind, question, answer_fn in QUESTION_SPECS:
            rows.append(
                {
                    "scene_id": scene_id,
                    "split": split,
                    "file_name": file_name,
                    "color": scene["color"],
                    "shape": scene["shape"],
                    "count": scene["count"],
                    "direction": scene["direction"],
                    "question_kind": question_kind,
                    "question": question,
                    "answer": answer_fn(scene),
                }
            )

    with metadata_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return pd.DataFrame(rows)


df = create_dataset(force=False)
print(df.shape)
display(df.head())
print(df.groupby(["split", "question_kind"]).size())
"""


VISUALIZE_DATASET = r"""
import matplotlib.pyplot as plt

examples = df.drop_duplicates("scene_id").sample(8, random_state=7)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, (_, row) in zip(axes.ravel(), examples.iterrows()):
    image = Image.open(IMAGE_DIR / row["file_name"]).convert("RGB")
    ax.imshow(image)
    ax.set_title(f"{row['color']} {row['shape']} | {row['count']} dots | {row['direction']}")
    ax.axis("off")
plt.tight_layout()
plt.show()
"""


FORMAT_DATASET = r"""
from PIL import Image

SYSTEM_MESSAGE = (
    "You are answering questions about synthetic glyph panels. "
    "Give the exact short answer requested. Do not explain."
)


def row_to_vlm_sample(row):
    image = Image.open(IMAGE_DIR / row["file_name"]).convert("RGB")
    return {
        "images": [image],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": row["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row["answer"]}],
            },
        ],
    }


train_rows = df[df["split"] == "train"].reset_index(drop=True)
eval_rows = df[df["split"] == "eval"].reset_index(drop=True)
train_dataset = [row_to_vlm_sample(row) for _, row in train_rows.iterrows()]
eval_dataset = [row_to_vlm_sample(row) for _, row in eval_rows.iterrows()]

print("Train samples:", len(train_dataset))
print("Eval samples:", len(eval_dataset))
print(train_dataset[0]["messages"][1]["content"][1]["text"])
print("Answer:", train_dataset[0]["messages"][2]["content"][0]["text"])
"""


MODEL_HELPERS = r"""
import gc
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dtype_for_device():
    if DEVICE != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_base_model():
    dtype = dtype_for_device()
    kwargs = {
        "torch_dtype": dtype,
        "_attn_implementation": "eager",
    }
    if DEVICE == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **kwargs)
    if DEVICE != "cuda":
        model = model.to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def generate_answer(model, processor, image, question, max_new_tokens=24):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = generated_ids[:, inputs["input_ids"].shape[-1]:]
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return answer.strip()


print("Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("dtype:", dtype_for_device())
"""


BASELINE_PREDICTIONS = r"""
model, processor = load_base_model()

probe_rows = pd.concat([
    train_rows[train_rows["question_kind"].isin(["glyph_code", "control_token"])].head(3),
    eval_rows[eval_rows["question_kind"].isin(["glyph_code", "control_token"])].head(3),
])

records = []
for _, row in probe_rows.iterrows():
    image = Image.open(IMAGE_DIR / row["file_name"]).convert("RGB")
    pred = generate_answer(model, processor, image, row["question"])
    records.append({
        "split": row["split"],
        "question_kind": row["question_kind"],
        "question": row["question"],
        "expected": row["answer"],
        "base_prediction": pred,
    })

baseline_df = pd.DataFrame(records)
display(baseline_df)
clear_memory()
"""


TRAINING_CODE = r"""
import torch
from peft import LoraConfig
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
USE_4BIT = False  # Keep False for the most predictable Colab Pro path. Set True if VRAM is tight.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE != "cuda":
    raise RuntimeError("This training notebook expects a GPU runtime. In Colab: Runtime -> Change runtime type -> GPU.")

major, _ = torch.cuda.get_device_capability()
USE_BF16 = major >= 8
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

print("GPU:", torch.cuda.get_device_name(0))
print("bf16:", USE_BF16)
print("adapter output:", ADAPTER_DIR)

model_kwargs = {
    "torch_dtype": TORCH_DTYPE,
    "device_map": "auto",
    "_attn_implementation": "eager",
}

if USE_4BIT:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=TORCH_DTYPE,
    )

model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **model_kwargs)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model.config.use_cache = False

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

training_args = SFTConfig(
    output_dir=str(ADAPTER_DIR),
    num_train_epochs=4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    optim="adamw_torch",
    bf16=USE_BF16,
    fp16=not USE_BF16,
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
    max_length=None,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,
)

trainer.train()
trainer.save_model(str(ADAPTER_DIR))
processor.save_pretrained(str(ADAPTER_DIR))

print("Saved LoRA adapter to:", ADAPTER_DIR)
"""


TRAINING_CHECK = r"""
from pathlib import Path

expected_files = ["adapter_config.json", "adapter_model.safetensors"]
for name in expected_files:
    path = ADAPTER_DIR / name
    print(name, "OK" if path.exists() else "MISSING", path)
"""


EVAL_CODE = r"""
import gc
import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not (ADAPTER_DIR / "adapter_config.json").exists():
    raise FileNotFoundError(f"No LoRA adapter found at {ADAPTER_DIR}. Run notebook 02 first.")


def dtype_for_device():
    if DEVICE != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model(with_adapter=False):
    kwargs = {
        "torch_dtype": dtype_for_device(),
        "_attn_implementation": "eager",
    }
    if DEVICE == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **kwargs)
    if with_adapter:
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    if DEVICE != "cuda":
        model = model.to(DEVICE)
    model.eval()
    return model


processor = AutoProcessor.from_pretrained(str(ADAPTER_DIR))


def generate_answer(model, image, question, max_new_tokens=24):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = generated_ids[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def normalize(text):
    return str(text).strip().lower().replace("assistant:", "").strip()


probe_rows = pd.concat([
    train_rows[train_rows["question_kind"].isin(["glyph_code", "control_token"])].head(6),
    eval_rows[eval_rows["question_kind"].isin(["glyph_code", "control_token"])].head(6),
]).reset_index(drop=True)

base_model = load_model(with_adapter=False)
base_preds = []
for _, row in probe_rows.iterrows():
    image = Image.open(IMAGE_DIR / row["file_name"]).convert("RGB")
    base_preds.append(generate_answer(base_model, image, row["question"]))
del base_model
clear_memory()

adapter_model = load_model(with_adapter=True)
adapter_preds = []
for _, row in probe_rows.iterrows():
    image = Image.open(IMAGE_DIR / row["file_name"]).convert("RGB")
    adapter_preds.append(generate_answer(adapter_model, image, row["question"]))
del adapter_model
clear_memory()

results = probe_rows[["split", "scene_id", "question_kind", "question", "answer"]].copy()
results["base_prediction"] = base_preds
results["adapter_prediction"] = adapter_preds
results["base_exact"] = [normalize(p) == normalize(a) for p, a in zip(base_preds, results["answer"])]
results["adapter_exact"] = [normalize(p) == normalize(a) for p, a in zip(adapter_preds, results["answer"])]

display(results)
print("Base exact match:", results["base_exact"].mean())
print("Adapter exact match:", results["adapter_exact"].mean())
"""


VISUAL_EVAL = r"""
sample = probe_rows.iloc[0]
image = Image.open(IMAGE_DIR / sample["file_name"]).convert("RGB")
plt.figure(figsize=(4, 4))
plt.imshow(image)
plt.axis("off")
plt.title(sample["question_kind"])
plt.show()

display(results[results["scene_id"] == sample["scene_id"]])
"""


cells_01 = [
    md(
        """
# 01 - Dataset and Baseline SmolVLM

This notebook creates a tiny synthetic visual question answering dataset and checks what the base SmolVLM model says before fine-tuning.

The dataset is deliberately artificial. The base model may understand colors, dots, and arrows, but it should not already know invented labels such as `luma-spire-3` or `mip-left`.
"""
    ),
    md(
        """
## What you will learn

- What a VQA fine-tuning row looks like.
- How images, questions, and answers are formatted for a VLM.
- Why a baseline matters before fine-tuning.
- How to save a small dataset to Google Drive for later notebooks.
"""
    ),
    code(SETUP_INSTALL),
    code(DRIVE_SETUP),
    code(SYNTHETIC_DATASET),
    code(VISUALIZE_DATASET),
    code(FORMAT_DATASET),
    md(
        """
## Load the base model

We use `HuggingFaceTB/SmolVLM-256M-Instruct` because it is small enough for fast Colab experiments. The model card describes SmolVLM as an image+text model that generates text, which fits VQA fine-tuning well.
"""
    ),
    code(MODEL_HELPERS),
    code(BASELINE_PREDICTIONS),
    md(
        """
## What to notice

The base model may answer simple visual questions, but it should struggle with made-up codes. That gap is exactly what notebook 02 trains the LoRA adapter to close.
"""
    ),
]


cells_02 = [
    md(
        """
# 02 - LoRA Fine-Tune SmolVLM

This notebook fine-tunes `HuggingFaceTB/SmolVLM-256M-Instruct` on the synthetic glyph VQA dataset using LoRA adapters.

The base model stays frozen. LoRA trains small adapter matrices and saves them to Google Drive.
"""
    ),
    md(
        """
## Runtime

Use a GPU runtime. Colab Pro with L4, A100, or V100 is plenty for this 256M model. T4 should also work, though it may be slower.
"""
    ),
    code(SETUP_INSTALL),
    code(DRIVE_SETUP),
    code(SYNTHETIC_DATASET),
    code(FORMAT_DATASET),
    md(
        """
## Fine-tuning setup

This follows the same basic pattern as the Hugging Face SmolVLM fine-tuning cookbook:

1. Load a VLM and processor.
2. Format each example as messages plus images.
3. Configure LoRA.
4. Train with TRL `SFTTrainer`.
5. Save the adapter.

Important VLM detail: `max_length=None` keeps training from accidentally truncating image tokens.
"""
    ),
    code(TRAINING_CODE),
    code(TRAINING_CHECK),
    md(
        """
## What got saved

The important files are:

- `adapter_config.json`
- `adapter_model.safetensors`

Those are the LoRA adapter. You still need the base SmolVLM model when you reload the adapter.
"""
    ),
]


cells_03 = [
    md(
        """
# 03 - Evaluate the Saved LoRA Adapter

This notebook reloads the base SmolVLM model, attaches the saved LoRA adapter, and compares before/after answers.
"""
    ),
    code(SETUP_INSTALL),
    code(DRIVE_SETUP),
    code(SYNTHETIC_DATASET),
    code(VISUALIZE_DATASET),
    code(FORMAT_DATASET),
    md(
        """
## Compare base model vs adapter

The exact-match score is intentionally simple. For this synthetic dataset, answers are short strings, so exact match is a useful first check.
"""
    ),
    code(EVAL_CODE),
    code(VISUAL_EVAL),
    md(
        """
## How to interpret results

- If adapter answers improve on `glyph_code` and `control_token`, the LoRA adapter learned the dataset-specific mapping.
- If train examples improve but eval examples do not, the adapter mostly memorized.
- If neither improves, increase epochs, raise the number of scenes, or inspect formatting before changing model size.
"""
    ),
]


def main():
    for old in NOTEBOOKS.glob("*.ipynb"):
        old.unlink()
    write_notebook("01_smolvlm_dataset_and_baseline.ipynb", cells_01)
    write_notebook("02_smolvlm_lora_finetune.ipynb", cells_02)
    write_notebook("03_evaluate_adapter.ipynb", cells_03)


if __name__ == "__main__":
    main()

