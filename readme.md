
---

# 🧠 Training Strategy: Phases, Domains, and Rewards (Tunix + Gemma)

This project trains a **Gemma3-1B + LoRA** model using **GRPO (Tunix)** with a **curriculum-based reinforcement learning strategy**.
The key idea is to **separate behavioral alignment from task competence**, and only introduce domain complexity after the model reliably follows the output contract.

---

## 🎯 Core Principle

> **Reinforcement Learning teaches behavior, not skills.**
> Skills come from the base model and data; RL enforces *how* the model responds.

Therefore:

* Early phases focus on **format, discipline, and attempt behavior**
* Later phases introduce **correctness**
* Domain diversity is introduced **only after behavior stabilizes**

---

## 🥇 Phase 1 — Behavioral Alignment (Format & Discipline)

### 🎯 Objective

Train the model to:

* Always attempt an answer
* Always output **only** inside:

  ```text
  <reasoning>...</reasoning>
  <answer>...</answer>
  ```
* Never refuse
* Avoid rambling or extraneous text

**Correctness is NOT optimized in this phase.**

---

### 📚 Dataset

* **GSM8K (subset)**
* Random 20–30% slice, or filtered for short questions

Reason:

* GSM8K prompts are short and structured
* Easy to parse
* Domain content is irrelevant at this stage

---

### 🏆 Reward Functions (Phase 1)

```python
reward_fns = [
    punish_refusal,        # hard negative for refusing / dodging
    strict_format_gate,    # requires valid <answer> tag
    light_length_penalty   # discourages rambling
]
```

**No correctness rewards.**
**No numeric checks.**
**No partial credit.**

Format is a **gate**, not a prize.

---

### 📊 Metrics to Track

* **Answer extraction rate** (primary KPI)
* Refusal rate
* Average output length

### ✅ Phase Completion Criteria

* ≥ 95% successful `<answer>` extraction
* Near-zero refusal
* Stable output length

---

## 🥈 Phase 2 — Correctness & Honesty (Single Domain)

### 🎯 Objective

Teach the model:

* Be correct when possible
* Be concise when wrong
* Do not hallucinate

---

### 📚 Dataset

* **Full GSM8K**

The domain stays fixed to avoid confounding behavior with task switching.

---

### 🏆 Reward Functions (Phase 2)

```python
reward_fns = [
    punish_refusal,
    strict_format_gate,
    check_answer_strict,    # exact match only (no ratios)
    penalize_length         # stronger penalty for verbose wrong answers
]
```

Design choices:

* Binary correctness signal
* No ratio-based partial credit
* Wrong answers are penalized, especially if verbose

---

### 📊 Metrics to Track

* Answer extraction rate (should remain high)
* Hallucination frequency
* Average length of wrong answers
* Accuracy trend (expected to improve slowly)

---

## 🟦 Phase 3 — Domain Generalization (Optional but Recommended)

### 🎯 Objective

Apply the learned **behavioral contract** across domains:

* Code
* Creative reasoning
* Science
* Open-ended QA

---

### 📚 Dataset

A **mixed dataset**, e.g.:

| Domain         | Proportion |
| -------------- | ---------- |
| GSM8K          | 50%        |
| Code           | 25%        |
| QA / Reasoning | 25%        |

---

### 🏆 Reward Functions (Phase 3)

```python
reward_fns = SAME AS PHASE 2
```

**Important rule:**

> **Do NOT change rewards when introducing new domains.**

Consistency ensures behavior generalizes.

---

### 📊 Metrics to Track

* Format compliance across domains
* Refusal rate
* Rambling frequency
* Qualitative reasoning quality

---

## 🔁 Phase Transitions (Rules)

* ❌ Do NOT change dataset and rewards at the same time
* ❌ Do NOT mix domains before Phase 1 stabilizes
* ✅ Only advance phases after behavior plateaus

---

## 🧠 Why This Works for Tunix Evaluation

* Enforces the exact output format required by judges
* Aligns with human + LLM-as-judge evaluation
* Avoids over-optimizing math (low eval weight)
* Produces consistent, interpretable reasoning traces

---

## 🏁 Summary

| Phase   | Focus               | Dataset       | Rewards                      |
| ------- | ------------------- | ------------- | ---------------------------- |
| Phase 1 | Format & discipline | GSM8K subset  | Format + refusal + brevity   |
| Phase 2 | Correctness         | Full GSM8K    | Strict correctness + brevity |
| Phase 3 | Generalization      | Mixed domains | Same as Phase 2              |

---

> **Final takeaway:**
> First teach the model *how to behave*.
> Only then ask it to be right — everywhere.

---

If you want next:

* 📈 a **phase auto-switching heuristic**
* 🧪 a **live extraction-rate logger**
* 🧠 a **judge-aligned evaluation prompt**

say the word 😤🔥
def iterable_slice(dataset, start_frac, end_frac):
    N = len(dataset)
    start = int(start_frac * N)
    end = int(end_frac * N)

    def generator():
        for i, sample in enumerate(dataset):
            if i >= end:
                break
            if i >= start:
                yield sample

    return generator()
