import re
import joblib
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
st.set_page_config(page_title="Smishing Detector", page_icon="🛡️")
 
W_LR, W_RF, W_DL = 0.2, 0.3, 0.5
THRESHOLD = 0.5
MAXLEN = 100
 
 
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("artifacts/tfidf.pkl")
    lr = joblib.load("artifacts/lr.pkl")
    rf = joblib.load("artifacts/rf.pkl")
    tokenizer = joblib.load("artifacts/tokenizer.pkl")
    dl = load_model("artifacts/bilstm.keras")
    return tfidf, lr, rf, tokenizer, dl
 
 
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()
 
 
def predict(text):
    tfidf, lr, rf, tokenizer, dl = load_artifacts()
    cleaned = clean_text(text)
    X = tfidf.transform([cleaned])
    seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=MAXLEN)
    p_lr = lr.predict_proba(X)[0, 1]
    p_rf = rf.predict_proba(X)[0, 1]
    p_dl = float(dl.predict(seq, verbose=0).flatten()[0])
    final = W_LR * p_lr + W_RF * p_rf + W_DL * p_dl
    return final, {"Logistic Regression": p_lr, "Random Forest": p_rf, "Bi-LSTM": p_dl}
 
 
st.title("🛡️ Smishing (SMS Phishing) Detector")
st.write("by Mariya Shaikh")
st.write("Paste an SMS message below to check whether it looks like smishing.")
 
message = st.text_area("SMS message", height=150,
                       placeholder="e.g. URGENT! Your bank account is blocked. Click now!")
 
if st.button("Analyze", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        with st.spinner("Analyzing..."):
            score, parts = predict(message)
        if score > THRESHOLD:
            st.error(f"⚠️ Smishing / Spam message (risk score: {score:.1%})")
        else:
            st.success(f"✅ Looks legitimate (risk score: {score:.1%})")
        st.progress(float(min(max(score, 0.0), 1.0)))
        with st.expander("See individual model scores"):
            for name, p in parts.items():
                st.write(f"**{name}**: {p:.1%}")
 
st.caption("Classifier trained on a limited dataset; it can make mistakes. "
           "Never click links or share OTPs/PINs from unexpected messages.")
with st.bottom:
    st.write("© copyright Mariya Shaikh. All rights reserved.")
