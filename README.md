# 📰 NewsLens — AI News Classifier

An NLP-powered web app that classifies news headlines into
**World, Sports, Business, or Sci/Tech** categories instantly.


## 🛠️ Tech Stack
- Python, Scikit-learn, TF-IDF, Logistic Regression
- Streamlit for UI
- AG News Dataset (120,000 articles)
- ~90% Classification Accuracy

## 📁 Project Structure
news-classifier/
├── model/
│   └── label_map.json
├── app.py
├── train.py
├── requirements.txt
└── README.md

## ⚙️ Run Locally
pip install -r requirements.txt
streamlit run app.py