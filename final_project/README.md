# Narrative-to-Scenario Crash Reconstruction Framework

This repository implements a narrative-driven crash reconstruction framework that converts natural-language crash narratives into executable CARLA simulation scenarios using lightweight language models, closed-loop optimization, and reward-guided trajectory refinement.

## Key Features

- Narrative-to-scenario crash reconstruction
- Lightweight LLM parameter initialization
- CARLA-based trajectory generation
- Simulator-in-the-loop refinement
- Reward-guided optimization
- Archetype-aware crash scoring
- Multi-view crash rendering

---

# Repository Pipeline

```text
Crash Inventory
      ↓
Crash Family Selection
      ↓
Pseudo-GT Parameter Generation
      ↓
CARLA Scenario Generation
      ↓
Trajectory / Outcome Extraction
      ↓
LLM Dataset Creation
      ↓
Few-Shot Parameter Prediction
      ↓
Feedback Refinement
      ↓
Candidate Search
      ↓
RL-Guided Optimization
      ↓
Multi-View Rendering
```

---

# Environment Setup

## Python Environment

```bash
conda create -n crash_recon python=3.10
conda activate crash_recon
```

Install dependencies:

```bash
pip install pandas numpy torch transformers openpyxl matplotlib
```

---

# CARLA Setup

Recommended:
- CARLA 0.9.15
- Town04 map

Launch CARLA:

```bash
./CarlaUE4.sh
```

The scripts expect CARLA server on:

```text
localhost:2000
```

---

# Experimental Workflow

## Step 1: Select Crash Cases

```bash
python scripts/01_select_25_cases.py
```

## Step 2: Generate Parameters

```bash
python scripts/02_auto_generate_params.py
python scripts/03_generate_carla_params.py
```

## Step 3: Run CARLA Simulations

```bash
python scripts/05_batch_run_carla.py
```

## Step 4: Create LLM Dataset

```bash
python scripts/07_create_llm_dataset.py
```

## Step 5: Run Few-Shot Prediction

```bash
python scripts/08_fewshot_predict_params.py
```

## Step 6: Evaluate Baseline

```bash
python scripts/09_evaluate_llm_vs_pseudogt.py
```

## Step 7: Feedback Refinement

```bash
python scripts/10_feedback_refine_llm_params.py
python scripts/11_evaluate_feedback_vs_pseudogt.py
```

## Step 8: Candidate Search

```bash
python scripts/12_feedback_candidate_search.py
python scripts/13_evaluate_feedback_vs_pseudogt.py
```

## Step 9: RL Optimization

```bash
python scripts/14_rl_search_refinement.py
python scripts/15_evaluate_rl_vs_pseudogt.py
```

## Step 10: Multi-View Rendering

```bash
python scripts/16_render_multiview_best_rl.py
```

---

# Notes

- CARLA must be running before simulation execution.
- GPU acceleration is recommended for LLM inference.
- The framework focuses primarily on Family A roadway-departure crash scenarios.
- This implementation is intended as a proof-of-concept research framework.

