# 📰 Fake News & Sentiment Detection System  
Full NLP Pipeline Implementation  

---

## 📌 Abstract  
This project implements an end-to-end pipeline for **fake news detection and sentiment analysis** using:  
- **Classical ML** (Naive Bayes, Logistic Regression, SVM)  
- **Deep Learning** (Bi-LSTM)  
- **Transformers** (DistilBERT, RoBERTa)  

On ~45k balanced news articles, the system achieved **91.2% accuracy with LSTM**, showing the evolution of NLP methods from TF-IDF to modern transformers.  

---

## 🚀 Workflow  
1. **Preprocessing** → text cleaning, stopwords removal, lemmatization  
2. **Classical ML** → TF-IDF + Naive Bayes, Logistic Regression, SVM  
3. **Deep Learning** → Bi-LSTM with embeddings  
4. **Transformers** → DistilBERT + sentiment analysis (RoBERTa)  
5. **Evaluation** → Accuracy, Precision, Recall, F1, Confusion Matrix  
6. **Deployment** → API-ready architecture  

---

## 📊 Results  

| Model                | Accuracy |
|-----------------------|----------|
| Naive Bayes           | 82.4%    |
| Logistic Regression   | 89.6%    |
| SVM                   | 88.3%    |
| **LSTM (best)**       | **91.2%**|
| Transformer-like      | 78.4%    |

- **Fake News** → more sensational, negative tone  
- **Real News** → balanced, factual tone  

---

## ⚖️ Ethical Considerations  
- **Bias:** dataset political/language bias possible  
- **Risk:** false positives may censor legitimate news  
- **Transparency:** explainable AI & human oversight required  

---

## 🛠️ Tech Stack  
- **Python** (Pandas, NumPy, Scikit-learn, Keras/TensorFlow, Transformers)  
- **Visualization:** Matplotlib, Seaborn, WordCloud  
- **Deployment:** REST API (Flask/FastAPI ready)  

---

### 📂 Repository Structure

```
.
├── notebook/ .ipynb file # Full pipeline for the project
├── requirements.txt # Lists all necessary dependencies
├── README.md # Provides an overview and project summary
├── data/ # Contains the dataset
└── visuals/ # Stores plots, word clouds, and other visual outputs
```


---

## 🔮 Future Work  
- Multimodal (text + images) fake news detection  
- Multi-language support  
- Real-time social media integration  

---


