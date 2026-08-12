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

def numeric_overlap(response, documents):
    resp_nums = set(re.findall(r'\d+\.?\d*', str(response)))
    doc_nums = set(re.findall(r'\d+\.?\d*', str(documents)))
    if not resp_nums:
        return 1.0  # no numbers claimed, nothing to hallucinate
    return len(resp_nums & doc_nums) / len(resp_nums)

def load_model():
    # Use relative path if running from the root of the project, else use absolute path
    model_path = os.path.join("Notebook", "Using_semantic_and_Tfidf", "save_model", "model.joblib")
    abs_model_path = r"C:\Users\BHUPATHI NADAR\OneDrive\Desktop\Main_project\MLSC-Task\Notebook\Using_semantic_and_Tfidf\save_model\model.joblib"
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return joblib.load(abs_model_path)

def load_tfidf_vectorizer():
    tfidf_path = os.path.join("Notebook", "Using_semantic_and_Tfidf", "save_model", "tfidf_vectorizer.joblib")
    abs_tfidf_path = r"C:\Users\BHUPATHI NADAR\OneDrive\Desktop\Main_project\MLSC-Task\Notebook\Using_semantic_and_Tfidf\save_model\tfidf_vectorizer.joblib"
    
    if os.path.exists(tfidf_path):
        return joblib.load(tfidf_path)
    elif os.path.exists(abs_tfidf_path):
        return joblib.load(abs_tfidf_path)
    else:
        print("TF-IDF vectorizer not found. Training one now... (this may take a minute on first run)")
        data_path = os.path.join("Data", "updated_data.csv")
        abs_data_path = r"C:\Users\BHUPATHI NADAR\OneDrive\Desktop\Main_project\MLSC-Task\Data\updated_data.csv"
        
        p = data_path if os.path.exists(data_path) else abs_data_path
        df = pd.read_csv(p)
        corpus = pd.concat([df['question'], df['documents'], df['response']]).astype(str)
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf.fit(corpus)
        joblib.dump(tfidf, abs_tfidf_path)
        return tfidf

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

def preprocess(question: str, document: list, answer: str):
    doc_str = str(document)
    
    # 1. Semantic embeddings
    st = ml_models["sentence_transformer"]
    q_emb = st.encode([question])
    d_emb = st.encode([doc_str])
    a_emb = st.encode([answer])
    
    doc_response_sim = cosine_similarity(d_emb, a_emb)[0][0]
    question_response_sim = cosine_similarity(q_emb, a_emb)[0][0]
    question_doc_sim = cosine_similarity(q_emb, d_emb)[0][0]
    
    # 2. TF-IDF features
    tfidf = ml_models["tfidf_vectorizer"]
    q_tfidf = tfidf.transform([question])
    d_tfidf = tfidf.transform([doc_str])
    a_tfidf = tfidf.transform([answer])
    
    tfidf_doc_response_sim = cosine_similarity(d_tfidf, a_tfidf)[0][0]
    tfidf_question_response_sim = cosine_similarity(q_tfidf, a_tfidf)[0][0]
    tfidf_question_doc_sim = cosine_similarity(q_tfidf, d_tfidf)[0][0]
    
    # 3. Numeric overlap
    num_overlap = numeric_overlap(answer, doc_str)
    
    # Ensure columns match training data order and names
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
