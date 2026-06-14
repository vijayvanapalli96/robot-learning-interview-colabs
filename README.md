# Robot Learning Interview Colab Notebooks

This repo is designed for 24-hour interview prep around VLMs, VLAs, hierarchy, action representation, and real robot deployment.

The goal is not to train a serious robot model. The goal is to make the ideas runnable, speakable, and easy to rehearse.

## Notebooks

1. `00_interview_map.ipynb`
   - Builds the compact mental model: VLM -> VLA -> policy -> robot deployment.
   - Produces a one-page interview map.

2. `01_vlm_to_vla_training_loop.ipynb`
   - Shows the conceptual move from image+text -> text to image+text+robot state -> action chunks.
   - Includes a tiny PyTorch-style skeleton.

3. `02_action_chunking_latency_tradeoffs.ipynb`
   - Simulates action chunk size tradeoffs.
   - Helps explain smoothness, responsiveness, jitter, and latency.

4. `03_hierarchy_ghost_wheelchair.ipynb`
   - Turns the GHOST hierarchy idea into a wheelchair-pushing task decomposition.
   - Produces subgoals, safety constraints, and evaluation hooks.

5. `04_policy_evaluation_failure_logging.ipynb`
   - Creates a tiny failure log and evaluation dashboard.
   - Trains the instinct to talk about success rate, interventions, latency, and failure modes.

6. `05_optional_nanovlm_code_reading.ipynb`
   - Optional code-reading notebook for nanoVLM-style VLM architecture.
   - Use only if setup stays clean.

## Suggested 24-hour sequence

Run `00`, `01`, `03`, and `04` first. Run `02` if you need more confidence discussing action chunks. Run `05` only if you have spare time.

## Opening in Colab

After pushing this repo to GitHub, open a notebook with:

```text
https://colab.research.google.com/github/<your-username>/<repo-name>/blob/main/notebooks/00_interview_map.ipynb
```

Replace the notebook filename as needed.

