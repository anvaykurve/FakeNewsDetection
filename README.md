# Fake News Detection using NLP

## About the Project

Misinformation is a growing problem in the digital age. This project aims to build an automated system that can detect fake news based on textual content. By analysing the linguistic patterns in news articles, we trained machine learning models to distinguish between authentic and fabricated stories.

### Key Features

- Text cleaning pipeline (Lemmatization, Stopword removal)  
- Comparison of Statistical (TF-IDF) vs. Semantic (Word2Vec) text representation  
- Implementation of Logistic Regression, SVM, and Random Forest classifiers  

---

## Dataset

The dataset used for this project contains labelled news articles with the following columns:

- `title`: The headline of the news article  
- `text`: The main body content  
- `label`: The target variable (e.g., 1 for Fake, 0 for Real)  

**Note:** The dataset is processed to combine title and text into a single content feature for better context.

---

## Installation



```bash
###Clone the repository

git clone https://github.com/anvaykurve/FakeNewsDetection.git
cd FakeNewsDetection

### Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

### Install dependencies

pip install -r requirements.txt

### If requirements.txt is missing, install the core libraries manually:

pip install pandas numpy scikit-learn nltk gensim matplotlib seaborn

### Download NLTK data
### Open a Python shell and run:

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

```

## Project Structure

## 📂 Project Structure

```bash
FakeNewsDetection/
│
├── data/                   # Dataset files (raw and cleaned)
├── notebooks/              # Jupyter/Colab notebooks for experiments
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Feature_Extraction_TFIDF.ipynb
│   ├── 03_Feature_Extraction_Word2Vec.ipynb
│   └── 04_Model_Training.ipynb
├── src/                    # Source code scripts (optional)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```
## 🚀 Methodology

### 1. Preprocessing

Raw text data is noisy. We implemented a cleaning pipeline that includes:

- Lowercasing: To ensure uniformity  
- Regex Cleaning: Removing URLs, special characters, and numbers  
- Stopword Removal: Eliminating common words (e.g., "the", "is") that add little value  
- Lemmatization: Converting words to their root form (e.g., "running" → "run")  

### 2. Feature Extraction

We experimented with two techniques to convert text into numerical vectors:

- **TF-IDF (Term Frequency–Inverse Document Frequency):** Captures the importance of words based on frequency  
- **Word2Vec:** A deep learning-based embedding technique that captures semantic relationships and context  

### 3. Models

We trained and evaluated the following supervised learning algorithms:

- Logistic Regression: A strong baseline for binary classification  
- Support Vector Machine (SVM): Effective in high-dimensional spaces  
- Random Forest: An ensemble method to reduce overfitting  

---

## 🛠 Technologies Used

- **Language:** Python 3.x  
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Gensim, Matplotlib, Seaborn  
- **Tools:** VSCode, Google Colab, Git  

---

Made by Anvay Kurve

