import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# Load data and model (cached for speed)
# ------------------------------------------------
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv("shl_assessments_full_details.csv")

    # Combine text for embeddings
    df["text"] = (
        df["Assessment Name"].fillna('') + ". " +
        df["Description"].fillna('') + " " +
        df["Job levels"].fillna('') + " " +
        df["Test Type"].fillna('')
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), normalize_embeddings=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return df, model, index


# ------------------------------------------------
# Initialize app
# ------------------------------------------------
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

st.title(" SHL Assessment Recommendation System")
st.write("Enter a job description or query to get the most relevant SHL assessments.")

df, model, index = load_data_and_model()

query = st.text_area("Enter your job description or query:", height=150, placeholder="e.g., Hiring a Python developer with good communication skills")
top_k = st.slider("Number of recommendations", 5, 10, 5)

if st.button("🔍 Get Recommendations"):
    if query.strip():
        query_vec = model.encode([query], normalize_embeddings=True)
        scores, ids = index.search(np.array(query_vec), top_k)

        results = df.iloc[ids[0]][
            ["Assessment Name", "URL", "Test Type", "Remote Testing", "Adaptive/IRT"]
        ].reset_index(drop=True)

        st.subheader("Top Recommendations")
        st.table(results)
    else:
        st.warning("Please enter a query before searching.")
