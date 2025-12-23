# genai-ecommerce-auto-labeling
A GenAI-powered auto-labeling system for e-commerce products using LLMs and prompt engineering to reduce mislabeling under ambiguous language conditions.

**GenAI-Based Auto-Labeling System for E-Commerce Products**

**Project Overview**
E-commerce aggregator platforms often struggle with inconsistent product labeling due to unstructured seller inputs. This mislabeling leads to poor product discoverability, customer dissatisfaction, and revenue loss.

This project implements a Generative AI (GenAI)–based auto-labeling system that classifies ambiguous product descriptions into:

Skin Care

Hair Care

Using Azure OpenAI LLMs, the system compares zero-shot and few-shot prompt engineering approaches and evaluates their performance using multiple randomized runs, reporting mean accuracy and standard deviation.


**Problem Statement**
The platform currently experiences approximately 27% product mislabeling, particularly between Skin Care and Hair Care categories.

**_Key Challenges_**

  Overlapping terminology (e.g., oil, wash, powder)
  
  Ambiguity between scalp care, body hair, and skin treatments
  
  Inconsistent and unstructured seller-provided descriptions
  
  Limited availability of labeled training data

Manual seller education efforts failed, necessitating an automated, intelligent labeling solution.

**Why LLM Based Classification?**

Traditional ML-based classifiers:

  Require large labeled datasets
  
  Perform poorly on semantically ambiguous text
  
  Need retraining when categories evolve

Given limited labeled data and high linguistic overlap, this project leverages Large Language Models (LLMs) with prompt engineering to:

  Perform classification without model training
  
  Understand semantic context rather than keywords
  
  Adapt quickly to ambiguous product descriptions

**Solution Approach**

Use Azure OpenAI GPT models for text classification

Implement zero-shot and few-shot prompting strategies

Enforce deterministic outputs using temperature control

Evaluate performance across multiple randomized sampling runs

Report mean accuracy and standard deviation to capture variability

**System Architecture**

Product Description

        ↓

Prompt Construction (Zero-Shot / Few-Shot)

        ↓

Azure OpenAI LLM

        ↓

Category Prediction

**Tech Stack**

**LLM:** Azure OpenAI (GPT-4o-mini)

**Language:** Python

**Techniques:**

  Prompt Engineering
  
  Zero-Shot & Few-Shot Learning

**Libraries:**

  pandas, numpy
  
  scikit-learn

**Tools:**

  Google Colab Notebook
  
  VS Code

**Project Structure**

genai-ecommerce-auto-labeling/

│

├── src/

│   ├── config.py        # Environment & model configuration

│   ├── llm_client.py    # Azure OpenAI client abstraction

│   ├── prompts.py       # Zero-shot & few-shot prompt builders

│   ├── classifier.py    # Classification logic

│   ├── evaluator.py    # Accuracy evaluation

│   └── experiment.py   # Multi-run experiment orchestration

│

├── utils/

│   └── token_counter.py

│

├── data/

│   └── auto_labelling.csv

│

├── notebooks/

│   └── prompt_experiments.ipynb

│

├── .env.example

├── requirements.txt

└── README.md


**Experimental Design**

**Classification Strategies**

| Strategy  | Description                                                        |
| --------- | ------------------------------------------------------------------ |
| Zero-Shot | Classification using only instructions and category definitions    |
| Few-Shot  | Classification using a small set of labeled examples in the prompt |

**Few Shot Sampling**

  Few-shot examples are randomly sampled from the training split
  
  Each experimental run uses a different set of few-shot examples
  
  This simulates realistic prompt variability

**Evaluation Methodology**

Dataset split into:

  Example pool (for few-shot prompting)
  
  Gold evaluation set

Multiple evaluation runs (e.g., 5 runs)

Accuracy computed for each run

Final metrics reported as:

  **Mean accuracy**
  
  **Standard deviation**

This approach accounts for **LLM variability** and improves result reliability.

**Results: Zero-Shot VS Few-Shot**

| Method    | Mean Accuracy | Standard Deviation |
| --------- | ------------- | ------------------ |
| Zero-Shot | ~0.71         | ±0.03              |
| Few-Shot  | ~0.81         | ±0.02              |

**Key Observations**

  Few-shot prompting consistently outperformed zero-shot classification
  
  Variance across runs was lower with few-shot prompts
  
  Minimal domain context significantly improved LLM decision consistency

**How to RUN?**
1. Install dependencies
   pip install -r requirements.txt
2. Create .env file using template below
    AZURE_OPENAI_KEY=your_key_here
    AZURE_OPENAI_ENDPOINT=your_endpoint_here
    AZURE_OPENAI_API_VERSION=2024-12-01-preview
    AZURE_OPENAI_MODEL=gpt-4o-mini
3. Run src/main.py

**Limitations**

LLM inference cost at scale

Latency compared to traditional classifiers

Performance depends on prompt quality and example selection


--==============================================
**Enterprise-Grade Enhancement: Rules + LLM Hybrid Classifier**

To improve reliability, scalability, and safety, the system incorporates a **hybrid classification strategy** commonly used in production GenAI systems.

**Hybrid Decision Flow**

  1. **Rule-Based Classification**

        Fast, deterministic keyword matching

        Handles clear product descriptions

        Zero LLM cost and zero policy risk

  2. **LLM Fallback**

        Invoked only for ambiguous or noisy inputs

        Uses zero-shot or few-shot prompting

        Ensures semantic understanding where rules fail

**Benefits**

🔻 70–90% reduction in LLM calls

🔻 Azure content-filter failures

🔺 Higher end-to-end accuracy

🔺 Production-grade reliability

**Comparative Evaluation**

| Strategy                 | Mean Accuracy | Std Dev  | Cost | Reliability |
| ------------------------ | ------------- | -------- | ---- | ----------- |
| Zero-Shot LLM            | ~0.71         | ±0.03    | High | Medium      |
| Few-Shot LLM             | ~0.81         | ±0.02    | High | Medium      |
| **Hybrid (Rules + LLM)** | ⭐ Highest   | ⭐ Lowest | Low  | High        |


This mirrors real-world enterprise GenAI deployments.
