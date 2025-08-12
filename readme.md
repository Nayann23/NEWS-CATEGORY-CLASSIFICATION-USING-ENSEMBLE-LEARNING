📰 News Classification App
🔍 Overview
This project implements a machine learning pipeline for classifying news articles into four categories: World, Sports, Business, and Sci/Tech. The solution leverages text data preprocessing, TF-IDF vectorization, and ensemble learning techniques to achieve high accuracy. A user-friendly Streamlit-based web app allows interactive predictions based on user input.

✨ Features
🧹 Preprocessing and vectorization of raw news text using TF-IDF.

🤖 Ensemble learning with stacking of Logistic Regression, Multinomial Naive Bayes, and LinearSVC models.

⚙️ Hyperparameter tuning using GridSearchCV to optimize model performance.

🖥️ Streamlit frontend for easy, real-time prediction with category mapping and prediction confidence.

🚀 Lightweight and efficient, suitable for deployment.

📚 Dataset
Dataset used: AG News Dataset (from Kaggle), which contains news articles categorized into World, Sports, Business, and Sci/Tech.

Training data size: 120,000 samples (30,000 per category).

Testing data size: 7,600 samples.

🛠️ Installation
# Clone the repository
git clone https://github.com/Nayann23/NEWS-CATEGORY-CLASSIFICATION-USING-ENSEMBLE-LEARNING.git
cd news-classification-app

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

🚀 Usage
Ensure stacking_model.pkl and tfidf.pkl are in the project directory (these files contain the trained model and vectorizer).

Run the Streamlit app:

streamlit run app.py

Open the displayed local URL in your browser.

Input news text and click Predict to see the predicted news category.

🗂️ Project Structure
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

📊 Model Details
Vectorization: TF-IDF with text preprocessing.

Base Models: Logistic Regression, Multinomial Naive Bayes, LinearSVC.

Ensemble Method: StackingClassifier with Logistic Regression as the meta-classifier.

Hyperparameter Tuning: GridSearchCV applied on base models.

Evaluation: Achieved approximately 92% accuracy on the test set.

🔮 Future Work
🌐 Extend to multi-lingual news classification.

🤖 Incorporate deep learning models like BERT for improved semantic understanding.

🔗 Build a REST API for easier integration with other applications.

📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

🙏 Acknowledgments
AG News dataset from Kaggle.

Streamlit for interactive UI.

Scikit-learn for machine learning algorithms and utilities.
