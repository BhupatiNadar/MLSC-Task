# MLSC-Task: Hallucination Detection in Generative AI

A machine learning pipeline that classifies LLM responses as **GROUNDED**, **PARTIALLY_GROUNDED**, or **HALLUCINATED** relative to their source documents, served through a FastAPI inference endpoint.

<p align="center">
  <img src="https://raw.githubusercontent.com/BhupatiNadar/MLSC-Task/HEAD/Images/Grounded.png" width="30%" alt="Grounded example"/>
  <img src="https://raw.githubusercontent.com/BhupatiNadar/MLSC-Task/HEAD/Images/Partially%20grounded.png" width="30%" alt="Partially grounded example"/>
  <img src="https://raw.githubusercontent.com/BhupatiNadar/MLSC-Task/HEAD/Images/Hallu.png" width="30%" alt="Hallucinated example"/>
</p>

<p align="center"><em>Left → Right: a fully grounded answer, a partially grounded answer, and a hallucinated answer.</em></p>

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Approach](#approach)
5. [Results](#results)
6. [Code Walkthrough](#code-walkthrough)
   - [`schemas.py`](#schemaspy)
   - [`train_tfidf.py`](#train_tfidfpy)
   - [`main.py` — FastAPI service](#mainpy--fastapi-service)
   - [Notebooks](#notebooks)
7. [Setup & Usage](#setup--usage)
8. [Roadmap](#roadmap)

---

## Overview

This repository detects **hallucinations** in Retrieval-Augmented Generation (RAG) systems — i.e. it checks whether an LLM's `response` is actually supported by the `document(s)` it was given, in the context of a `question`. Every example is labeled with one of three classes:

| Label | Meaning |
|---|---|
| `GROUNDED` | ✅ The response is fully supported by the source documents. |
| `PARTIALLY_GROUNDED` | ⚠️ Some claims are supported, others are not. |
| `HALLUCINATED` | ❌ The response is unsupported by / contradicts the documents. |

<p align="center">
  <img src="https://raw.githubusercontent.com/BhupatiNadar/MLSC-Task/HEAD/Images/Hallu.png" width="60%" alt="Hallucination illustration"/>
</p>

---

## Repository Structure

```
MLSC-Task/
├── Images/                          # Diagrams used in this README
├── Notebook/
│   ├── Tfidf/
│   │   └── svm.ipynb                # TF-IDF + LinearSVC baseline & SMOTE experiment
│   ├── Semantic/
│   │   ├── semantic_features.ipynb          # Embedding + similarity feature extraction (local/CPU)
│   │   ├── semantic_features_updaedData.ipynb # Same, on Colab/GPU with the updated dataset
│   │   └── load_updaed_data.ipynb           # Reassembles cached .npy embeddings into a DataFrame
│   └── Using_semantic_and_Tfidf/
│       └── save_model/
│           ├── model.joblib             # Final trained classifier
│           └── tfidf_vectorizer.joblib  # Fitted TF-IDF vectorizer
├── schemas.py                        # Pydantic request/response models for the API
├── train_tfidf.py                    # Standalone script to fit & persist the TF-IDF vectorizer
├── main.py                           # FastAPI inference service
├── requirements.txt / pyproject.toml # Dependencies
└── README.md
```

---

## Dataset

The dataset is the `hotpotqa` subset of **`rungalileo/ragbench`** (loaded via HuggingFace `datasets`), containing `question`, `documents`, `response`, and a `label` column.

Class distribution is heavily imbalanced:

- **Grounded** ≈ 47,265
- **Hallucinated** ≈ 40,397
- **Partially Grounded** ≈ 7,515

This imbalance is why **SMOTE** oversampling is used before training (see [`svm.ipynb`](#notebooks)).

---

## Approach

Two feature-engineering paths were explored:

1. **Lexical (TF-IDF)** — concatenate `question` + `documents` + `response`, vectorize with `TfidfVectorizer(max_features=10000, ngram_range=(1,2))`.
2. **Semantic (Sentence Embeddings)** — encode `question`, `documents`, `response` separately with `all-MiniLM-L6-v2`, then compute pairwise cosine similarities between them.

At inference time (`main.py`), **both** feature families plus a numeric-overlap heuristic are combined into a single 7-feature vector fed into the saved classifier.

---

## Results

| Model | Features | Macro-F1 | Accuracy |
|---|---|---|---|
| Logistic Regression | TF-IDF | 0.32 | — |
| Logistic Regression | Semantic + SMOTE | 0.589 | 65.5% |
| LinearSVC | Semantic + SMOTE | 0.595 | 65.9% |
| **LinearSVC** | **TF-IDF** | **0.66** | — |

🏆 **Champion model:** `LinearSVC` trained on TF-IDF features — currently the strongest of the tested configurations.

---

## Code Walkthrough

### `schemas.py`

Defines the request/response contracts for the API using **Pydantic**, so FastAPI can validate incoming JSON automatically and generate OpenAPI docs.

```python
from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    question: str
    document: List[str]
    answer: str

class PredictResponse(BaseModel):
    label: str
    confidence: float
```

| Line | Explanation |
|---|---|
| `class PredictRequest(BaseModel):` | Defines the shape of the JSON body the `/predict` endpoint expects. |
| `question: str` | The user's question — a single string. |
| `document: List[str]` | The retrieved source passages — a **list** of strings (multiple documents can support one answer). |
| `answer: str` | The LLM's generated response that needs to be checked for grounding. |
| `class PredictResponse(BaseModel):` | Defines the shape of the JSON the API returns. |
| `label: str` | One of `GROUNDED`, `PARTIALLY_GROUNDED`, `HALLUCINATED`. |
| `confidence: float` | The model's predicted probability for the chosen label (0–1). |

---

### `train_tfidf.py`

A standalone utility script that fits a TF-IDF vectorizer on the full corpus and saves it to disk, so `main.py` doesn't need to refit it every time it loads.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def main():
    print("Loading data...")
    df_1 = pd.read_csv(r"...\Data\updated_data.csv")
    print("Concatenating corpus...")
    corpus = pd.concat([df_1['question'], df_1['documents'], df_1['response']]).astype(str)
    print("Fitting TfidfVectorizer...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_vectorizer.fit(corpus)
    print("Saving TfidfVectorizer...")
    joblib.dump(tfidf_vectorizer, r"...\save_model\tfidf_vectorizer.joblib")
    print("Done!")

if __name__ == "__main__":
    main()
```

| Line | Explanation |
|---|---|
| `df_1 = pd.read_csv(...)` | Loads the raw training data (`updated_data.csv`) into a DataFrame. |
| `corpus = pd.concat([df_1['question'], df_1['documents'], df_1['response']]).astype(str)` | Stacks the three text columns into **one long Series** so the vectorizer learns vocabulary from questions, documents, *and* responses combined, not just one field. `.astype(str)` guards against non-string/NaN values. |
| `TfidfVectorizer(max_features=5000, stop_words='english')` | Builds a vectorizer capped at the 5,000 most informative terms and strips common English stop-words. |
| `tfidf_vectorizer.fit(corpus)` | Learns the vocabulary and IDF weights from the combined corpus (no transform yet — this only fits). |
| `joblib.dump(tfidf_vectorizer, ...)` | Serializes the fitted vectorizer to disk so it can be reloaded instantly at inference time without retraining. |
| `if __name__ == "__main__":` | Ensures `main()` only runs when the script is executed directly (`python train_tfidf.py`), not when imported. |

---

### `main.py` — FastAPI service

The production inference server. It loads the trained classifier, the sentence-embedding model, and the TF-IDF vectorizer once at startup, then exposes a `/predict` endpoint.

```python
import os
import re
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from schemas import PredictRequest, PredictResponse

LABELS = ["GROUNDED", "PARTIALLY_GROUNDED", "HALLUCINATED"]
ml_models = {}
```

| Line | Explanation |
|---|---|
| `from contextlib import asynccontextmanager` | Used to define FastAPI's modern startup/shutdown lifecycle hook (`lifespan`). |
| `LABELS = [...]` | The canonical class order used elsewhere for reference/documentation. |
| `ml_models = {}` | A module-level dict that acts as a simple in-memory cache for the loaded model, embedder, and vectorizer, avoiding global reloads per request. |

#### Numeric overlap heuristic

```python
def numeric_overlap(response, documents):
    resp_nums = set(re.findall(r'\d+\.?\d*', str(response)))
    doc_nums = set(re.findall(r'\d+\.?\d*', str(documents)))
    if not resp_nums:
        return 1.0  # no numbers claimed, nothing to hallucinate
    return len(resp_nums & doc_nums) / len(resp_nums)
```

| Line | Explanation |
|---|---|
| `re.findall(r'\d+\.?\d*', str(response))` | Regex extracts every number (integer or decimal) mentioned in the response. |
| `set(...)` | Deduplicates the numbers so overlap is computed on unique values, not counts. |
| `if not resp_nums: return 1.0` | If the response cites **no numbers at all**, there's nothing numeric to hallucinate, so the feature defaults to a "perfect" score. |
| `len(resp_nums & doc_nums) / len(resp_nums)` | Set intersection ÷ total numbers claimed = fraction of the response's numeric claims that actually appear in the source documents. Directly penalizes fabricated statistics. |

#### Model loading helpers

```python
def load_model():
    model_path = os.path.join("Notebook", "Using_semantic_and_Tfidf", "save_model", "model.joblib")
    abs_model_path = r"...\model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return joblib.load(abs_model_path)
```

| Line | Explanation |
|---|---|
| `os.path.join(...)` | Builds a **relative** path so the app works when launched from the project root on any machine. |
| `abs_model_path` | A hard-coded fallback (developer's local path) used only if the relative path isn't found — useful during local debugging, not portable. |
| `if os.path.exists(...)` | Prefers the portable relative path first; falls back only when necessary. |

```python
def load_tfidf_vectorizer():
    ...
    else:
        print("TF-IDF vectorizer not found. Training one now...")
        df = pd.read_csv(p)
        corpus = pd.concat([df['question'], df['documents'], df['response']]).astype(str)
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf.fit(corpus)
        joblib.dump(tfidf, abs_tfidf_path)
        return tfidf
```

| Line | Explanation |
|---|---|
| Checks relative → absolute path, same pattern as `load_model()` | Two-tier fallback for portability. |
| **Self-healing fallback:** if no saved vectorizer is found anywhere, it **retrains one on the fly** from `Data/updated_data.csv` using the exact same logic as `train_tfidf.py`, so the API never hard-fails just because a `.joblib` file is missing. |

#### App startup (`lifespan`)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    ml_models["model"] = load_model()
    ml_models["sentence_transformer"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ml_models["tfidf_vectorizer"] = load_tfidf_vectorizer()
    print("Models loaded successfully!")
    yield
    ml_models.clear()

app = FastAPI(title="Hallucination Classifier API", lifespan=lifespan)

@app.get("/")
def health():
    return {"status": "ok"}
```

| Line | Explanation |
|---|---|
| `@asynccontextmanager` + `lifespan(app)` | FastAPI's recommended pattern for **startup/shutdown** logic — everything before `yield` runs once when the server boots, everything after runs on shutdown. |
| `ml_models["model"] = load_model()` | Loads the trained `LinearSVC` classifier into memory **once**, not per-request (expensive to reload). |
| `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` | Loads the same embedding model used during training, ensuring feature parity between train and inference. |
| `ml_models["tfidf_vectorizer"] = load_tfidf_vectorizer()` | Loads (or trains) the TF-IDF vectorizer used for the lexical similarity features. |
| `yield` | Hands control back to FastAPI to start serving requests; code after `yield` runs on shutdown. |
| `ml_models.clear()` | Frees memory on shutdown. |
| `app = FastAPI(..., lifespan=lifespan)` | Registers the lifespan hook with the app instance. |
| `@app.get("/")` → `health()` | A simple liveness/health-check endpoint for uptime monitoring or container orchestration probes. |

#### Feature engineering (`preprocess`)

```python
def preprocess(question: str, document: list, answer: str):
    doc_str = str(document)

    st = ml_models["sentence_transformer"]
    q_emb = st.encode([question])
    d_emb = st.encode([doc_str])
    a_emb = st.encode([answer])

    doc_response_sim = cosine_similarity(d_emb, a_emb)[0][0]
    question_response_sim = cosine_similarity(q_emb, a_emb)[0][0]
    question_doc_sim = cosine_similarity(q_emb, d_emb)[0][0]

    tfidf = ml_models["tfidf_vectorizer"]
    q_tfidf = tfidf.transform([question])
    d_tfidf = tfidf.transform([doc_str])
    a_tfidf = tfidf.transform([answer])

    tfidf_doc_response_sim = cosine_similarity(d_tfidf, a_tfidf)[0][0]
    tfidf_question_response_sim = cosine_similarity(q_tfidf, a_tfidf)[0][0]
    tfidf_question_doc_sim = cosine_similarity(q_tfidf, d_tfidf)[0][0]

    num_overlap = numeric_overlap(answer, doc_str)

    features = pd.DataFrame([{
        'doc_response_similarity': doc_response_sim,
        'question_response_similarity': question_response_sim,
        'question_document_similarity': question_doc_sim,
        'tfidf_doc_response_sim': tfidf_doc_response_sim,
        'tfidf_question_response_sim': tfidf_question_response_sim,
        'tfidf_question_document_sim': tfidf_question_doc_sim,
        'numeric_overlap': num_overlap
    }])
    return features
```

| Line | Explanation |
|---|---|
| `doc_str = str(document)` | The incoming `document` field is a **list** of strings; this collapses it into one string so it can be embedded/vectorized as a single unit. |
| `st.encode([question])` / `[doc_str]` / `[answer]` | Runs each text field through the sentence-transformer to get a dense embedding vector. Wrapped in lists because `.encode()` expects a batch. |
| `cosine_similarity(d_emb, a_emb)[0][0]` | Semantic similarity between **documents and the answer** — the core "is the answer grounded in the source?" signal. |
| `cosine_similarity(q_emb, a_emb)[0][0]` | Semantic similarity between **question and answer** — checks the answer is actually on-topic. |
| `cosine_similarity(q_emb, d_emb)[0][0]` | Semantic similarity between **question and documents** — checks retrieval quality (not directly about hallucination, but useful context). |
| `tfidf.transform([question])` etc. | Same three fields converted into TF-IDF sparse vectors using the **already-fitted** vectorizer (`.transform`, not `.fit_transform`, since fitting must only happen once, during training). |
| `tfidf_doc_response_sim`, etc. | The lexical (word-overlap) counterparts of the three semantic similarities above — captures literal wording overlap that embeddings can miss. |
| `num_overlap = numeric_overlap(answer, doc_str)` | Adds the numeric-fact-checking heuristic described earlier. |
| `features = pd.DataFrame([{...}])` | Packs all **7 engineered features** into a single-row DataFrame with named columns — the exact same column names/order the classifier was trained on, which is critical for correct predictions. |

#### Inference & endpoint

```python
def run_inference(model, features):
    probs = model.predict_proba(features)[0]
    pred_class = model.predict(features)[0]
    confidence = np.max(probs)
    return str(pred_class), float(confidence)

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        features = preprocess(request.question, request.document, request.answer)
        label, confidence = run_inference(ml_models["model"], features)
        return PredictResponse(label=label, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

| Line | Explanation |
|---|---|
| `model.predict_proba(features)[0]` | Gets the class-probability distribution for the single input row. |
| `model.predict(features)[0]` | Gets the hard predicted label. |
| `confidence = np.max(probs)` | Uses the highest class probability as the reported confidence score. |
| `@app.post("/predict", response_model=PredictResponse)` | Registers the `POST /predict` route; FastAPI validates the response against `PredictResponse` automatically. |
| `request: PredictRequest` | FastAPI automatically parses & validates the incoming JSON body against the `PredictRequest` schema. |
| `try / except HTTPException(500, ...)` | Any failure in preprocessing or inference is caught and returned as a clean HTTP 500 with the error message, instead of crashing the server. |

---

### Notebooks

<p align="center">
  <img src="https://raw.githubusercontent.com/BhupatiNadar/MLSC-Task/HEAD/Images/Partially%20grounded.png" width="55%" alt="Feature engineering pipeline"/>
</p>

- **`Tfidf/svm.ipynb`** — Splits the data (80/20, stratified), builds TF-IDF features (`ngram_range=(1,2)`, `max_features=10000`), trains a `LinearSVC(class_weight="balanced")`, evaluates with `classification_report` + confusion matrix, then repeats after applying **SMOTE** to compare balanced vs. unbalanced training.
- **`Semantic/semantic_features.ipynb`** — CPU-based reference run: loads raw data, embeds `question`/`documents`/`response` with `all-MiniLM-L6-v2`, computes the three pairwise cosine similarities, and writes an enriched CSV.
- **`Semantic/semantic_features_updaedData.ipynb`** — The same embedding pipeline executed on **Google Colab with a GPU**, on the larger `updated_data.csv`; saves embeddings to `.npy` files on Google Drive for reuse instead of recomputing every time.
- **`Semantic/load_updaed_data.ipynb`** — Loads the cached `.npy` embedding arrays back into a DataFrame, attaches them to the raw labels, recomputes cosine similarities, and exports `updated_semantic_data.csv` for model training.

---

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Pre-fit the TF-IDF vectorizer
python train_tfidf.py

# 3. Launch the API
uvicorn main:app --reload

# 4. Call the endpoint
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What is the capital of France?",
        "document": ["Paris is the capital and most populous city of France."],
        "answer": "The capital of France is Paris."
      }'
```

Example response:

```json
{
  "label": "GROUNDED",
  "confidence": 0.94
}
```

---

## Roadmap

- [ ] Train **XGBoost / LightGBM** on the semantic features to capture non-linear interactions.
- [ ] Fine-tune a **Cross-Encoder** (`DeBERTa-v3-small` / `RoBERTa-large`) formatted as `[CLS] Document [SEP] Response [SEP]` for direct NLI-style factual-consistency scoring.
- [ ] Explore **LLM-as-a-Judge** evaluation using an open-source instruct model.
- [ ] Add confidence thresholding → route low-confidence predictions to human review / `UNKNOWN`.
- [ ] Containerize with Docker and add drift monitoring for production deployment.

---

*Built as part of the MLSC Task — hallucination detection for trustworthy RAG systems.*