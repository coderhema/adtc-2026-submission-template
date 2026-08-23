# Technical Report - Hekima AI: Offline Multilingual Math & Scientific-Reasoning Assistant

**Team ID:** hekima
**Domain:** math_scientific_reasoning
**Model:** tiny-aya-global-q4_k_m (3.3B, Cohere2 architecture)
**Languages:** Yoruba, Hausa, Swahili, Igbo, English
**Working name:** Hekima (Swahili: "wisdom")

---

## Problem

Every year millions of African secondary students sit national examinations - WAEC and GCE in Nigeria, Ghana, Sierra Leone and The Gambia, NECO and JAMB/UTME in Nigeria, KCSE in Kenya, and many more across the continent. Most of them prepare with three scarce resources: teachers, textbooks, and electricity for cloud-connected devices. In rural schools a single teacher often covers several subjects; private lessons are out of reach for most families; and AI tutors that exist are English-only, cloud-dependent, and trained on US curriculum content.

**Hekima AI** is a fully offline, on-device math and scientific-reasoning tutor that solves past-paper questions step by step in the student's own language - Yoruba, Hausa, Swahili, Igbo, or English - on the 8 GB integrated-GPU laptops that are already common across the continent. A student types or pastes a math or science question (or asks for practice from a topic), and the model explains the solution methodically, the way a patient teacher would. No cloud, no API fees, no connectivity required.

## Design Decisions

- **Base model:** Cohere **Tiny Aya Global** (3.3B, Cohere2 architecture, 500K context, 67-language tokenizer). Selected because it natively speaks 9+ African languages - including Yoruba, Hausa, Swahili, Igbo, Amharic, Wolof, Zulu - which no comparably sized English-centric model (Qwen 2.5 3B, Llama 3.2 3B, Gemma 3 4B) can do without extensive custom fine-tuning.
- **Fine-tuning:** QLoRA (PEFT) + TRL SFTTrainer on curated exam QA data in English / Yoruba / Hausa / Swahili / Igbo. Cohere2 is not supported by Unsloth, so we use the standard PEFT stack, which fully supports the Cohere2 architecture. Trained on a free GPU tier (Colab T4).
- **Training data (verified available on Hugging Face):**
  - `masakhane/afrimgsm` - GSM8k math word problems translated into 16+ African languages, including Yoruba, Hausa, Swahili and Igbo. The core math-reasoning corpus.
  - `masakhane/afrimmlu` - MMLU translated into Yoruba, Hausa, Swahili, Igbo and English; broad math + science reasoning across all five target languages.
  - `cais/mmlu` (STEM subjects) - deep English math + science coverage (high-school and college level).
  - `openai/gsm8k` - extra English math word problems for deeper arithmetic reasoning.
  - `worldboss/waec-integrated-science-2007` - real WAEC past questions.
  - `honourjesus/nllb-hausa-waec-translations` - WAEC content in Hausa.
  - Curated bilingual QA pairs (English / Yoruba / Hausa / Swahili / Igbo) generated from WAEC, NECO, GCE and everyday math/science patterns; Yoruba, Hausa and Igbo flagged for native-speaker validation.
- **Quantization:** Q4_K_M (2.14 GB) - the largest quantization that comfortably fits the 7 GB peak-RAM budget alongside the app runtime, leaving headroom that maximizes the efficiency score (S_eff = 100 x (7 - peak) / 7).
- **App layer:** Mojo (Modular) - chat UI + prompt pre-processing. The profiler measures only the llama.cpp inference process; the application stack is unmeasured, so Mojo lets us ship a responsive UX without sacrificing score.
- **Alternatives considered and rejected:**
  - Qwen 2.5 3B / Llama 3.2 3B / Gemma 3 4B - excellent English/tech models, but African-language capability is near zero; rebuilding Yoruba+Hausa+Swahili via fine-tuning is not feasible in the time available.
  - Aya Expanse 8B - stronger multilingual model, but 8B Q4 is too slow and RAM-heavy on CPU; too risky against the 7 GB budget.
  - Unsloth for training - rejected because cohere2 is not in Unsloth's supported architecture list; PEFT/TRL is the supported path.

## Alignment with the Scoring Model

- **S_acc (50%):** the automated half (arc_easy) is grade-school science MCQs. Exam-prep fine-tuning directly targets this genre, and the judge-scored half evaluates responses to domain prompts - exactly what the tutor does. The qualitative story is strong: a model that explains a percentage-profit question in Yoruba, step by step, is immediately demonstrable to judges.
- **African Alpha bonus (+15%):** math word problems in Yoruba, Hausa, Swahili and Igbo are the training data itself; the bonus is robust across any language judges try live.
- **S_eff (20%):** Q4_K_M leaves ~4-5 GB RAM headroom for a high efficiency score.

## Constraints

- Target: 8 GB RAM, Intel i5 10th-12th gen / Ryzen 5, integrated GPU only, Ubuntu 22.04 LTS, 256 GB SSD.
- No GPU acceleration - pure CPU inference via llama.cpp (GGUF weights only, the sole permitted runtime).
- 100% offline during evaluation - no outbound network calls once profiling starts.
- Peak RAM <= 7 GB (hard limit; OOM = disqualification).
- No thermal throttling; core temp must stay <= 85 C (penalty: -10 points).

## Benchmarks

| Metric | Value |
|---|---|
| Machine | (dev laptop - to be measured) |
| Model size (Q4_K_M) | 2.14 GB |
| RAM at peak | to be measured |
| Time to first token | to be measured |
| Generation speed | to be measured (target >= 15 t/s reference) |
| Thermal throttling | None observed (to be confirmed) |

Self-reported development benchmarks. Official scores are measured by the ADTC profiler on the standard evaluation machine.

---

## Repository layout

- `metadata.json` - submission metadata, 2 test prompts (EN + YO)
- `download_model.sh` - fetches tiny-aya-global-q4_k_m.gguf from Hugging Face
- `model/` - weights land here (gitignored)
- `notebooks/` - QLoRA fine-tuning pipeline (Colab-ready)
