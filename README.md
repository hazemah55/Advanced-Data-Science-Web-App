# Advanced Data Science Web Application

## Project Overview

An interactive Data Science application developed using Python and Streamlit. The application combines Machine Learning, Natural Language Processing, and image processing into a single interface.

## Features

### 1. Spam Detection
- Classifies user-entered messages as spam or not spam.
- Uses a saved Machine Learning model and text vectorizer.

### 2. Text Summarization
- Supports English and Arabic text.
- Uses the Edmundson extractive summarization algorithm.
- Allows users to select summary length and specify bonus or stigma words.

### 3. Image Denoising
- Allows users to upload images.
- Supports resizing and rotation.
- Applies median filtering using OpenCV to reduce image noise.

### 4. Mushroom Classification
- Uses a trained Keras model to classify mushroom records.
- Accepts mushroom characteristics through interactive dropdown menus.
- Uses saved categorical encoders to prepare model inputs.

The mushroom classifier is an educational demonstration and must not be used to determine whether a real mushroom is safe to eat.

## Technologies Used

Python, Streamlit, Pandas, NumPy, NLTK, Sumy, OpenCV, TensorFlow, Keras, Scikit-learn, Pillow.

## Project Structure

```text
Advanced-Data-Science-Web-App/
├── Spam.py
├── spam.pk
├── vectorizer.pk
├── mushroom_model.h5
├── encoders.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

1. Clone or download the repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Start the application:

```bash
streamlit run Spam.py
```

4. Open the local Streamlit page and select a module from the sidebar.

## Future Improvements

- Add automated tests.
- Improve error handling and model loading.
- Add evaluation metrics and dataset documentation.
- Deploy the application using Streamlit Community Cloud.

## Author

**Hazem Ahmed Wageh**

Computer Science Student | Aspiring Data Scientist

[GitHub Profile](https://github.com/hazemah55)
