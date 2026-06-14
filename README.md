# SmolVLM LoRA Fine-Tuning Colab Lab

This repo contains three Colab notebooks for learning how vision-language model fine-tuning works with SmolVLM, LoRA adapters, PEFT, and TRL.

The notebooks use a tiny synthetic visual question answering dataset. The dataset is intentionally unusual: it contains generated glyph panels with colored shapes, marker dots, arrows, and made-up answer codes. That makes it easier to see the difference between the base model and the fine-tuned adapter.

## Notebooks

1. `01_smolvlm_dataset_and_baseline.ipynb`
   - Creates the synthetic VQA dataset.
   - Saves images and metadata to Google Drive.
   - Loads the base SmolVLM model.
   - Runs baseline predictions before fine-tuning.

2. `02_smolvlm_lora_finetune.ipynb`
   - Loads or regenerates the dataset.
   - Fine-tunes `HuggingFaceTB/SmolVLM-256M-Instruct` with LoRA adapters.
   - Saves the adapter to Google Drive.

3. `03_evaluate_adapter.ipynb`
   - Reloads the base model and the saved LoRA adapter.
   - Compares base-model outputs against adapter outputs.
   - Builds a small before/after results table.

## Colab Links

After pushing to GitHub, open:

```text
https://colab.research.google.com/github/vijayvanapalli96/robot-learning-interview-colabs/blob/main/notebooks/01_smolvlm_dataset_and_baseline.ipynb
```

Replace the notebook name for notebooks `02` and `03`.

## Notes

- These notebooks favor reliability over maximum speed.
- They avoid `flash-attn` because it can be fragile in Colab.
- The default model is the small `HuggingFaceTB/SmolVLM-256M-Instruct`.
- Adapter outputs are saved under `MyDrive/smolvlm_lora_lab`.

## Sources

- Hugging Face SmolVLM model card: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
- Hugging Face SmolVLM fine-tuning cookbook: https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl
- TRL SFTTrainer docs: https://huggingface.co/docs/trl/en/sft_trainer

