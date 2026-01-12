# 🧠 Multi-Domain GRPO Training with Phase-Based Curriculum (R1-Style)

This repository implements a **multi-domain Group Relative Policy Optimization (GRPO)** training pipeline inspired by **DeepSeek-R1 / Open-R1**, built fully in Python (no YAML), and designed for **small-model RL stability**.

The core ideas are:

* **Mixed-domain training** (math, code, science, logic)
* **Domain-aware reward scaling**
* **Phase-based reward enabling (curriculum RL)**
* **Strict format + correctness + conciseness enforcement**
* **LoRA-based policy learning with KL control**

This setup is intentionally minimal, debuggable, and notebook-friendly.

---

## 📦 Datasets Used

We train on a **small but diverse mixture** of high-signal datasets.

| Domain  | Dataset    | Size (Train) | Purpose              |
| ------- | ---------- | ------------ | -------------------- |
| Math    | GSM8K      | ~7.5k        | Numerical reasoning  |
| Code    | MBPP       | ~374         | Program synthesis    |
| Science | ARC-Easy   | ~2.2k        | Scientific reasoning |
| Logic   | StrategyQA | ~1.6k        | Commonsense / logic  |

### Why these?

* All are **verifiable or semi-verifiable**
* Small enough to iterate fast
* High signal-to-noise for RL
* Mirrors Open-R1’s early curriculum philosophy

---

## 🔀 Dataset Mixing Strategy

Datasets are mixed **at sampling time**, not concatenated.

```python
weights = {
  "math": 0.35,
  "code": 0.30,
  "science": 0.20,
  "logic": 0.15,
}
```

### What this means

* Each batch is **heterogeneous**
* Domains appear in proportion, not blocks
* Prevents overfitting to a single reasoning style
* Encourages general reasoning behavior

Each sample carries a `domain` field used later for reward scaling.

---

## 🧠 Model Architecture

* **Base model**: Gemma-3 1B Instruct
* **Policy**: LoRA-adapted actor
* **Reference**: Frozen base model
* **Optimization**: GRPO with KL regularization

Only LoRA parameters are updated.

---

## 🎯 Reward Functions

Rewards are **additive**, **domain-scaled**, and **phase-gated**.

### 🚨 Hard Negative

**`punish_refusal`**

* Nukes refusals, empty answers, and “I can’t help”
* Prevents RL collapse
* Active in *all phases*

---

### 🚪 Format Gate (Strict)

**`match_format_exactly`**

* Requires:

  ```xml
  <reasoning>...</reasoning>
  <answer>...</answer>
  ```
* Blocks learning without structure
* High weight early

---

### 🧭 Soft Format

**`match_format_approximately`**

* Partial credit for near-correct tagging
* Encourages compliance without brittleness

---

### 🎯 Correctness

**`check_answer`**

* Exact match → full reward
* Near match → partial
* Numeric tolerance → fractional reward
* Disabled early, enabled later

---

### 🔥 Termination Pressure

**`penalize_length_and_rambling`**

* Penalizes:

  * Overlong outputs
  * Talking after `</answer>`
  * Rambling phrases
* Enforces **short, sharp answers**
* Enabled only in final phase

---

## 🧮 Domain-Based Reward Scaling

Each reward is scaled by domain difficulty:

```python
DOMAIN_REWARD_SCALE = {
  "math": 1.0,
  "code": 1.2,
  "science": 0.8,
  "logic": 0.7,
}
```

### Why?

* Code is harder → stronger signal
* Science & logic are noisier → softer pressure
* Prevents reward imbalance across domains

---

## 🧩 Phase-Based Curriculum (R1-Style)

Training is split into **three phases** based on global step count.

### Phase 0 — Bootstrap (0 → 25%)

**Goal**: Make the model respond correctly *at all*

Enabled:

* Refusal penalty
* Strict format
* Soft format

Disabled:

* Correctness
* Length penalty

---

### Phase 1 — Correctness (25% → 75%)

**Goal**: Learn to be right

Enabled:

* Refusal penalty
* Format rewards
* Answer correctness

Disabled:

* Length penalty

---

### Phase 2 — Polish (75% → 100%)

**Goal**: Be concise, strict, and accurate

Enabled:

* All rewards
* Strong termination pressure

---

### Why phases?

* Early correctness rewards destabilize GRPO
* Format must be learned before correctness
* Length penalties too early cause truncation
* Mirrors DeepSeek-R1 training dynamics

---

## 🧠 Reward Routing Logic

A **RewardRouter** dynamically:

1. Detects current phase
2. Enables/disables rewards
3. Applies domain scaling
4. Returns a single GRPO-compatible score

This replaces static reward lists and enables **true curriculum RL**.

---

## ⚙️ GRPO Configuration Highlights

| Parameter         | Value | Why                          |
| ----------------- | ----- | ---------------------------- |
| `num_generations` | 4     | Group diversity              |
| `beta` (KL)       | 0.08  | Prevents policy drift        |
| `epsilon`         | 0.2   | PPO-style stability          |
| `temperature`     | 0.9   | Exploration                  |
| `max_tokens`      | 512   | Enough for reasoning         |
| `MAX_STEPS`       | fixed | Phase schedule depends on it |

⚠️ **MAX_STEPS should NOT be changed once phases are defined**.

---

## 🧪 What This Setup Gives You

✅ Stable GRPO
✅ Multi-domain reasoning
✅ Curriculum learning
✅ Format compliance
✅ Concise answers
✅ Debuggable rewards
✅ Notebook-friendly iteration

This is **not toy RL** — it’s a faithful, practical reproduction of modern reasoning-RL techniques.

---

## 🚀 Extensions (Easy to Add)

* Code execution rewards (MBPP)
* Per-domain KL coefficients
* Phase transitions based on eval metrics
* Automatic dataset annealing
* Reward logging per domain

---

## 🏁 Final Note

This setup is intentionally **small-scale but correct**.
If it works here, it scales.

--- 

## References

- https://huggingface.co/learn/cookbook/trl_grpo_reasoning_advanced_reward
- https://github.com/huggingface/open-r1?tab=readme-ov-file#training-models
- 
