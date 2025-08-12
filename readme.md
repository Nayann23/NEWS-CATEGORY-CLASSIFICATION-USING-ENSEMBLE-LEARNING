# 📰 News Classification App

## 🔍 Overview

This project implements a machine learning pipeline for classifying news articles into four categories: World, Sports, Business, and Sci/Tech.  
It leverages text preprocessing, TF-IDF vectorization, and ensemble learning techniques to achieve high accuracy.  
A user-friendly Streamlit-based web app allows interactive predictions based on user input.

---

## ✨ Features

- 🧹 Preprocessing and vectorization of raw news text using TF-IDF.  
- 🤖 Ensemble learning with stacking of Logistic Regression, Multinomial Naive Bayes, and LinearSVC models.  
- ⚙️ Hyperparameter tuning with GridSearchCV for optimal model performance.  
- 🖥️ Streamlit frontend for real-time prediction with category mapping and prediction confidence.  
- 🚀 Lightweight and efficient — suitable for deployment.

---

## 📚 Dataset

- Dataset: AG News Dataset from Kaggle — news articles categorized into World, Sports, Business, and Sci/Tech.  
- Training Data: 120,000 samples (30,000 per category).  
- Testing Data: 7,600 samples.

---

## 🛠️ Installation


# Clone the repository
```
git clone https://github.com/Nayann23/NEWS-CATEGORY-CLASSIFICATION-USING-ENSEMBLE-LEARNING.git
```
---
# (Optional) Create and activate a virtual environment
```
python -m venv venv  
source venv/bin/activate    # On Windows: venv\Scripts\activate
```
---
# Install dependencies
```
pip install -r requirements.txt
```

---

## 🚀 Usage

Ensure that `stacking_model.pkl` and `tfidf.pkl` are in your project directory — (these files contain the trained model and TF-IDF vectorizer).  

Run the Streamlit app:

```
streamlit run app.py
```

Open the displayed local URL in your browser.  
Input a news text and click Predict to see the category.

---

## 🗂️ Project Structure

```
news-classification-app/  
│  
├── app.py  
├── NewsClassifierPro.ipynb  
├── stacking_model.pkl  
├── tfidf.pkl  
├── train.csv  
├── requirements.txt  
├── LICENSE  
└── README.md
```

---

## 📊 Model Details

- Vectorization: TF-IDF with text preprocessing.  
- Base Models: Logistic Regression, Multinomial Naive Bayes, LinearSVC.  
- Ensemble Method: StackingClassifier with Logistic Regression as meta-classifier.  
- Hyperparameter Tuning: Using GridSearchCV on base models.  
- Evaluation: Achieved ~92% accuracy on the test set.

---

## 🔮 Future Work

- 🌐 Extend to multi-lingual news classification.  
- 🤖 Integrate deep learning models like BERT for better semantic understanding.  
- 🔗 Develop a REST API for seamless integration with other apps.

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## 🙏 Acknowledgments

- AG News Dataset from Kaggle.  
- Streamlit for interactive UI.  
- Scikit-learn for ML algorithms and utilities.
```

Do you want me to also make a downloadable `README.md` file with all these lines bolded so you can paste it directly into your repo? That way you don’t have to copy-paste manually.
