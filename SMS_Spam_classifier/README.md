# SMS Spam Classifier

This project is a simple SMS/Email spam classifier built using Python and Streamlit. It uses a Naive Bayes model trained on a dataset of SMS messages to classify messages as spam or not spam.

## Features

- Preprocesses text data using tokenization, stopword removal, and stemming.
- Uses TF-IDF vectorization for feature extraction.
- Classifies messages using a Multinomial Naive Bayes model.
- Provides a Streamlit web interface for easy interaction.
- Includes a script to retrain the model on the provided dataset.

## Installation

1. Clone the repository or download the project files.

2. Install the required Python packages using pip:

```
pip install -r requirements.txt
```

3. Download the necessary NLTK data packages:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

### Running the Streamlit App

To start the spam classifier web app, run:

```
streamlit run app.py
```

Enter your message in the text area and click the "Predict" button to see if the message is classified as spam or not.

### Retraining the Model

If you want to retrain the model with the dataset (`spam.csv`), run:

```
python retrain_model.py
```

This will preprocess the data, train a new Naive Bayes model, and save the model and vectorizer as `model.pkl` and `vectorizer.pkl` respectively.

## Files

- `app.py`: Streamlit app for spam classification.
- `retrain_model.py`: Script to retrain the Naive Bayes model.
- `spam.csv`: Dataset used for training.
- `model.pkl`: Saved trained model (generated after retraining).
- `vectorizer.pkl`: Saved TF-IDF vectorizer (generated after retraining).

## License

This project is open source and free to use.
