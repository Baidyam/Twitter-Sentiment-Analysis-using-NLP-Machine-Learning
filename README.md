# End-to-End Twitter Sentiment Analysis using NLP & Machine Learning

## Project Overview
This project implements a complete Natural Language Processing (NLP) pipeline to perform binary sentiment analysis on Twitter data. The goal is to convert raw, noisy tweet text into meaningful numerical representations and train multiple machine learning models to accurately classify sentiment.

The project follows an industry-style workflow, starting from raw data preprocessing to feature extraction, model training, and evaluation.

## Dataset
- **Source:** [https://www.kaggle.com/datasets/kazanova/sentiment140]
- **Data Type:** Text (Tweets)
- **Target Variable:** Binary sentiment labels
  - `1` → Positive sentiment
  - `2` → Negative sentiment

## Tech Stack
- **Language:** Python
- **Libraries:**
  - `pandas`, `numpy`
  - `nltk`
  - `scikit-learn`
  - `matplotlib`, `seaborn`

## Project Workflow

### 1️⃣ Data Loading & Inspection
- Loaded tweet dataset into a pandas DataFrame
- Checked:
  - Dataset shape
  - Column names
  - Missing values
  - Class distribution

### 2️⃣ Text Cleaning & Preprocessing
Applied multiple NLP preprocessing techniques to clean noisy Twitter text:
- Converted text to lowercase
- Removed:
  - URLs
  - User mentions (`@user`)
  - Hashtags (`#`)
  - Punctuation
  - Numbers
  - Special characters
- Removed extra whitespaces

### 3️⃣ Tokenization
- Split cleaned tweets into individual tokens (words)
- Prepared tokenized text for further linguistic processing

### 4️⃣ Stopword Removal
- Removed common English stopwords using NLTK
- Reduced noise and improved feature quality

### 5️⃣ Lemmatization
- Applied WordNet Lemmatizer
- Converted words to their base forms (e.g., `running → run`, `better → good`)
- Helped reduce vocabulary size while preserving meaning

### 6️⃣ Text Reconstruction
- Recombined processed tokens back into clean text
- Final cleaned text used as model input

### 7️⃣ Feature Extraction using TF-IDF
- Converted text into numerical vectors using TF-IDF Vectorizer
- Captured:
  - Term importance
  - Frequency across documents
- Prevented dominance of commonly occurring words

### 8️⃣ Train-Test Split
- Split dataset into:
  - **Training set** (80%)
  - **Testing set** (20%)
- Ensured unbiased model evaluation

### 9️⃣ Machine Learning Models
Trained and evaluated multiple supervised learning models:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear model for binary classification |
| **Multinomial Naive Bayes** | Probabilistic classifier based on Bayes' theorem |
| **Support Vector Machine (SVM)** | Finds optimal hyperplane for classification |

Each model was trained using the same TF-IDF features for fair comparison.

### 🔟 Model Evaluation
Evaluated models using:
- **Accuracy Score**
- **Classification Report:**
  - Precision
  - Recall
  - F1-score

Compared model performance to identify the most effective classifier.

## Results Summary
- TF-IDF proved effective for representing tweet text
- Logistic Regression and SVM delivered strong classification performance
- Clean preprocessing significantly improved model accuracy
- Classical NLP techniques remain powerful for sentiment analysis tasks


## Future Improvements
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add real-time tweet sentiment analysis
- [ ] Create web application for sentiment prediction
- [ ] Extend to multi-class sentiment classification

## Author
**Moumita Baidya**  
Email: baidya.m@northeastern.edu
*Master's in Data Science | NLP & Machine Learning Enthusiast*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/yourusername)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/yourusername)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- NLTK team for NLP tools
- Scikit-learn developers
- Twitter for providing the dataset
