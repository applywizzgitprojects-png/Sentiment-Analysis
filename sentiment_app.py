import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ------------------------------------------------------------------
# One-time NLTK resource (safe to call every run; cached by NLTK)
# ------------------------------------------------------------------
nltk.download("vader_lexicon", quiet=True)
SIA = SentimentIntensityAnalyzer()

# Small handpicked positive lexicon for our negation override
POS_WORDS = {
    "love","lovely","amazing","awesome","great","good","nice","excellent","fantastic",
    "wonderful","superb","best","brilliant","enjoy","like","happy","satisfied","positive"
}
NEGATION_TOKENS = {
    "not","no","never","n't","dont","don't","cant","can't","cannot","won't","wont",
    "isn't","arent","aren't","wasn't","werent","weren't","didn't","doesn't","ain't"
}

# ------------------------------------------------------------------
# Cleaning & sentiment helpers
# ------------------------------------------------------------------
def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # keep words and apostrophes
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def vader_label(compound: float, pos_thresh: float = 0.05, neg_thresh: float = -0.05) -> str:
    if compound >= pos_thresh:
        return "positive"
    if compound <= neg_thresh:
        return "negative"
    return "neutral"

def negation_override(tokens: list[str], base_compound: float) -> float:
    """
    If a negation occurs within 3 tokens before a positive word, push
    the sentiment negative (or at least toward negative).
    This helps with: "I don't love this", "not good", etc.
    """
    # early-out if clearly negative already
    if base_compound <= -0.25:
        return base_compound

    for i, tok in enumerate(tokens):
        if tok in NEGATION_TOKENS:
            # check a small window of words that follow the negation
            window = tokens[i + 1 : i + 4]
            if any(w in POS_WORDS for w in window):
                # down-shift strongly; enough to cross the -0.05 threshold
                return base_compound - 0.7
    return base_compound

def analyze_text(text: str) -> tuple[float, str]:
    text = text if isinstance(text, str) else ""
    # VADER base score
    scores = SIA.polarity_scores(text)
    compound = scores["compound"]

    # Negation-aware override
    toks = simple_tokenize(text)
    compound = negation_override(toks, compound)

    return compound, vader_label(compound)

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def bar_counts(series: pd.Series, title: str):
    order = ["positive", "neutral", "negative"]
    counts = series.value_counts()
    for k in order:  # ensure all 3 appear
        counts.loc[k] = counts.get(k, 0)
    counts = counts[order]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax,
                palette=["#4caf50", "#9e9e9e", "#f44336"])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title(title)

    ymax = max(1, counts.max())
    ax.set_ylim(0, ymax * 1.18)
    for i, v in enumerate(counts.values):
        ax.text(i, v + ymax * 0.03, str(int(v)), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="wide")
    st.title("Sentiment Analysis App")
    st.caption("Single Analysis and Bulk Analysis shown together. No training path required.")

    # ---------------- Single Analysis ----------------
    st.subheader("Single Analysis")

    if "single_pred" not in st.session_state:
        st.session_state.single_pred = None

    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    with c1:
        text = st.text_input("Enter text:", value="I don't love this product, it’s amazing!")
    with c2:
        if st.button("Predict", type="primary", use_container_width=True):
            _, label = analyze_text(text)
            st.session_state.single_pred = label

    if st.session_state.single_pred is not None:
        st.success(f"**Predicted Sentiment:** {st.session_state.single_pred}")
    else:
        st.info("Enter text and click **Predict**.")

    st.markdown("---")

    # ---------------- Bulk Analysis ----------------
    st.subheader("Bulk Analysis")
    st.caption("Upload a CSV with a **text** column (or choose which column contains text). "
               "The table and bar chart appear immediately.")

    up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if up is not None:
        df = pd.read_csv(up)
        if df.empty:
            st.error("The uploaded CSV looks empty.")
            return

        cols = list(df.columns)
        default_text_col = "text" if "text" in cols else cols[0]
        text_col = st.selectbox("Choose the text column", options=cols,
                                index=cols.index(default_text_col))

        # Predict
        texts = df[text_col].astype(str).fillna("")
        compounds, labels = [], []
        for t in texts:
            c, l = analyze_text(t)
            compounds.append(c)
            labels.append(l)

        df["compound"] = compounds
        df["predicted_sentiment"] = labels

        st.dataframe(df, use_container_width=True)

        bar_counts(df["predicted_sentiment"], title="Sentiment Distribution (Uploaded File)")
    else:
        st.info("Upload a CSV to see the table and chart here.")

if __name__ == "__main__":
    main()
