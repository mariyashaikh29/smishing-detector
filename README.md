# 🛡️ Smishing (SMS Phishing) Detector

A web app that checks whether an SMS message looks like **smishing** (SMS phishing) or a legitimate message. It uses an ensemble of three machine learning models and is built with Streamlit.

**Live demo:** https://smishing-detector-mariya.streamlit.app

---

## How it works

1. The message is cleaned (links, numbers and punctuation removed, lowercased).
2. It is passed to three models:
   - **Logistic Regression** (TF-IDF features, unigrams + bigrams)
   - **Random Forest** (200 trees, TF-IDF features)
   - **Bi-LSTM** (Keras, word embeddings)
3. Their probabilities are combined with a weighted average:

   `risk score = 0.2 × LR + 0.3 × RF + 0.5 × Bi-LSTM`

4. If the risk score is above **0.5**, the message is flagged as smishing.

The app also shows each model's individual score.

## Training summary

- Duplicates removed, text cleaned, 80/20 stratified train/test split
- TF-IDF (5,000 features) fitted on training data only
- SMOTE applied on the training set to balance classes
- Bi-LSTM trained with early stopping
- Reported accuracy on the test set: about 0.98 (Logistic Regression), 0.99 (Random Forest), 0.99 (Bi-LSTM), 0.99 (Ensemble)

The full training code is in the Colab notebook (`Smishing_Detection_final_project.ipynb`).

## Project structure

```
.
├── app.py              # Streamlit web app
├── requirements.txt    # Pinned dependencies
├── README.md
└── artifacts/          # Trained models
    ├── tfidf.pkl
    ├── lr.pkl
    ├── rf.pkl
    ├── tokenizer.pkl
    └── bilstm.keras
```

## Run locally

Use **Python 3.13** (TensorFlow 2.20 has no wheels for Python 3.14 yet).

```bash
git clone https://github.com/<your-username>/smishing-detector.git
cd smishing-detector
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Deployment

The app is hosted on **Streamlit Community Cloud**. When deploying, choose **Python 3.13** under *Advanced settings* so the pinned TensorFlow version installs correctly.

## Limitations

- The models were trained on a limited dataset, so they can make mistakes on message styles they have not seen.
- Very high test accuracy does not guarantee the same performance on real-world messages.
- Always be careful with unexpected messages: never click unknown links or share OTPs and PINs.

## Author

**Mariya Shaikh**

© Mariya Shaikh. All rights reserved.
