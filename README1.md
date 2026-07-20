# SMS Spam Detection using Naive Bayes

A simple Machine Learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using the **Multinomial Naive Bayes** algorithm and **TF-IDF Vectorization**. The project also includes a simple Streamlit web application for testing custom messages.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit
- Joblib

## Dataset

- SMS Spam Collection Dataset

## Project Structure

```
spam-detection-ibm/
│
├── dataset/
├── models/
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

## Installation & Running

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python train.py
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

## Features

- SMS text preprocessing
- TF-IDF feature extraction
- Multinomial Naive Bayes classifier
- Model evaluation
- Interactive Streamlit interface for prediction