import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def main():
    print("Loading data...")
    df_1 = pd.read_csv(r"C:\Users\BHUPATHI NADAR\OneDrive\Desktop\Main_project\MLSC-Task\Data\updated_data.csv")
    print("Concatenating corpus...")
    corpus = pd.concat([df_1['question'], df_1['documents'], df_1['response']]).astype(str)
    print("Fitting TfidfVectorizer...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_vectorizer.fit(corpus)
    print("Saving TfidfVectorizer...")
    joblib.dump(tfidf_vectorizer, r"C:\Users\BHUPATHI NADAR\OneDrive\Desktop\Main_project\MLSC-Task\Notebook\Using_semantic_and_Tfidf\save_model\tfidf_vectorizer.joblib")
    print("Done!")

if __name__ == "__main__":
    main()
