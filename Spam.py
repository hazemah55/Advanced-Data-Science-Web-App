import streamlit as st  
import pandas as pd      
import pickle                  
import nltk 
import re   
import cv2
import numpy as np                         
from nltk.stem import WordNetLemmatizer   
from nltk.corpus import stopwords                     
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from PIL import Image
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
tokenizer = Tokenizer("english")
tokenizer1 = Tokenizer("arabic")

def apply_theme(theme="dark_green_gold🔰"):
    if theme == "dark_green_gold🔰":
        st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: white; }
        h1, h2, h3, h4 { color: forestgreen !important; }
        section[data-testid="stSidebar"] { background-color: #2c241b; }
        div.stButton > button {
            background-color: forestgreen; color: white; border-radius: 10px;
        }
        div.stButton > button:hover {
            background-color: darkgoldenrod; color: black;
        }
        </style>
        """, unsafe_allow_html=True)


    elif theme == "modern_purple💜":
        st.markdown("""
        <style>
        .stApp { background-color: #1a001a; color: #f0e6f6; }
        h1, h2, h3, h4 { color: #bb86fc !important; }
        section[data-testid="stSidebar"] { background-color: #2d0033; }
        div.stButton > button {
            background-color: #7b2cbf; color: white; border-radius: 15px;
        }
        div.stButton > button:hover {
            background-color: #9d4edd; color: black;
        }
        </style>
        """, unsafe_allow_html=True)


st.sidebar.title("App Settings")
theme_choice = st.sidebar.selectbox(
    "Choose Theme", 
    ["dark_green_gold🔰", "modern_purple💜"]
)
apply_theme(theme_choice)



model_choice= st.sidebar.selectbox("choose your model",('Home🏠','Spam_detection⚠️','Text_summarizer🔠','Image_editing🖼️','Mashroom_pred🍄'))

if model_choice == 'Home🏠':
    st.title("Welcome to my Advanced Data Science Final Project 📊📈")
    st.write("By: Hazem ahmed wageh")
    st.header('This app covers a Data Science Project developed using Streamlit,' 
    ' focusing on four key modules: Spam Detection, Text Summarization, Image Denoising, and Mushroom Edibility prediction')


if model_choice == 'Spam_detection⚠️':
    def loaded_model():
        with open(r'spam.pk', 'rb') as file:
            return pickle.load(file)
    def vectorizer_msg():
        with open(r'vectorizer.pk','rb') as file: 
            return pickle.load(file)
    
    def cleaning(text_filed):
        return text_filed
    model = loaded_model()
    vectorizer = vectorizer_msg()

    st.title('Streamlit Spam Detection')
    st.header('Welcome to the Spam Detection Model')

    user_message = st.text_area('Enter your message')

    if st.button("Predict"):
        if not user_message.strip():
            st.warning("Please enter a message first.")
        else:
            cleaned_message = cleaning(user_message)
            user_message_vec = vectorizer.transform([cleaned_message])
            prediction = model.predict(user_message_vec)[0]

            if prediction == 0:  
               st.error(" This message is SPAM!")
            else:
               st.success("This message is NOT spam.")


if model_choice == 'Text_summarizer🔠':
    
    choice = st.selectbox("Choose an language",["English🔤","Arabic"])

    if choice == "English🔤" :
        
        def edu(text, sentence_count=2, bonus_words=None, stigma_words=None):
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = EdmundsonSummarizer(Stemmer("english"))
            summarizer.null_words = get_stop_words("english")
            if bonus_words:
                summarizer.bonus_words = bonus_words
            if stigma_words:
                summarizer.stigma_words = stigma_words
            
            summary = summarizer(parser.document, sentence_count)
            return summary        


        st.title("Edmundson Summarizer")
        User_text = st.text_area("Enter the your text to summarizer",height=200)
        sentence_count = st.slider("Number of sentences in summary:", 1, 10, 2)
        bonus_words_inp  = st.text_input("Enter bonus words")
        stigma_words_inp = st.text_input("Enter stigma words")

        if st.button("Press the buttom for summarize"):
            if User_text.strip():
                bonus_words = [w.strip() for w in bonus_words_inp.split(",") if w.strip()]
                stigma_words = [w.strip() for w in stigma_words_inp.split(",") if w.strip()]
                summary = edu(User_text, sentence_count, bonus_words, stigma_words)
                st.subheader("Summary")
                for sentence in summary:
                    st.write(str(sentence))
            else:
                st.warning("Please enter some text to summarize.")

    elif choice == "Arabic":
        def edu(text, sentence_count=2, bonus_words=None, stigma_words=None):
            parser = PlaintextParser.from_string(text, Tokenizer("arabic"))
            summarizer = EdmundsonSummarizer(Stemmer("arabic"))
            summarizer.stop_words = get_stop_words("arabic")
            summarizer.null_words = get_stop_words("arabic")
            if bonus_words:
              summarizer.bonus_words = bonus_words
            if stigma_words:
                summarizer.stigma_words = stigma_words
            
            summary = summarizer(parser.document, sentence_count)
            return summary        


        st.title("ملخص إدموندسون")
        User_text = st.text_area("أدخل النص المراد تلخيصه",height=200)
        sentence_count = st.slider("عدد الجُمل في الملخص:", 1, 10, 2)
        bonus_words_inp  = st.text_input("أدخل الكلمات الإيجابية")
        stigma_words_inp = st.text_input("أدخل الكلمات المرفوضة")

        if st.button("اضغط على الزر للتلخيص"):
            if User_text.strip():
                bonus_words = [w.strip() for w in bonus_words_inp.split(",") if w.strip()]
                stigma_words = [w.strip() for w in stigma_words_inp.split(",") if w.strip()]
                summary = edu(User_text, sentence_count, bonus_words, stigma_words)
                st.subheader("Summary")
                for sentence in summary:
                    st.write(str(sentence))



if model_choice == 'Image_editing🖼️':
    st.title("Image Denoising")
    user_image =st.file_uploader("upload your image",type=['jpg','jpeg','png'])
    if user_image is not None:
        file_bytes = np.asarray(bytearray(user_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        max_width = st.slider("Resize image width (px)", min_value=100, max_value=1000, value=500, step=50)

        height, width = image.shape[:2]
        if width > max_width:
            scale = max_width / width
            new_size = (max_width, int(height * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        angle = st.slider("Rotate image (degrees)", min_value=-180, max_value=180, value=0, step=5)
        if angle != 0:
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))    

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.subheader("Original Image")
        st.image(image_rgb, caption="Noisy Image", use_container_width=True)
        if st.button("Denoise Image"):
            denoised_image = cv2.medianBlur(image, 5)
            denoised_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
            st.subheader("Denoised Image")
            st.image(denoised_rgb, caption="Denoised Image", use_container_width=True)
    else:
        st.info("Please upload an image to continue.")



if model_choice == 'Mashroom_pred🍄':
    model = load_model("mushroom_model.h5")
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    le = encoders["label_encoder"]  
    ohe = encoders["onehot_encoder"]

    st.title("🍄 Mushroom Classifier App")
    st.write("Select mushroom characteristics to predict if it’s **Edible (e)** or **Poisonous (p)**")

    feature_options = { 
        "cap-shape": {"b": "Bell", "c": "Conical", "x": "Convex", "f": "Flat", "k": "Knobbed", "s": "Sunken"},
        "cap-surface": {"f": "Fibrous", "g": "Grooves", "y": "Scaly", "s": "Smooth"},
        "cap-color": {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "r": "Green", "p": "Pink",
                      "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"},
        "bruises": {"t": "Bruises", "f": "No"},
        "odor": {"a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy", "f": "Foul",
                 "m": "Musty", "n": "None", "p": "Pungent", "s": "Spicy"},
        "gill-attachment": {"a": "Attached", "d": "Descending", "f": "Free", "n": "Notched"},
        "gill-spacing": {"c": "Close", "w": "Crowded", "d": "Distant"},
        "gill-size": {"b": "Broad", "n": "Narrow"},
        "gill-color": {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "g": "Gray", "r": "Green",
                       "o": "Orange", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"},
        "stalk-shape": {"e": "Enlarging", "t": "Tapering"},
        "stalk-root": {"b": "Bulbous", "c": "Club", "u": "Cup", "e": "Equal", "z": "Rhizomorphs",
                       "r": "Rooted", "?": "Missing"},
        "stalk-surface-above-ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
        "stalk-surface-below-ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
        "stalk-color-above-ring": {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange",
                                   "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"},
        "stalk-color-below-ring": {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange",
                                   "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"},
        "veil-type": {"p": "Partial", "u": "Universal"},
        "veil-color": {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"},
        "ring-number": {"n": "None", "o": "One", "t": "Two"},
        "ring-type": {"c": "Cobwebby", "e": "Evanescent", "f": "Flaring", "l": "Large", "n": "None",
                      "p": "Pendant", "s": "Sheathing", "z": "Zone"},
        "spore-print-color": {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "r": "Green",
                              "o": "Orange", "u": "Purple", "w": "White", "y": "Yellow"},
        "population": {"a": "Abundant", "c": "Clustered", "n": "Numerous", "s": "Scattered",
                       "v": "Several", "y": "Solitary"},
        "habitat": {"g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths", "u": "Urban",
                    "w": "Waste", "d": "Woods"}
    }

    
    user_input = {}
    for feature, options in feature_options.items():
        choice = st.selectbox(feature, list(options.values()))
        inv_map = {v: k for k, v in options.items()}
        user_input[feature] = inv_map[choice]

    input_df = pd.DataFrame([user_input])

    
    for col in ohe.feature_names_in_:
        if col not in input_df.columns:
            try:
                first_cat = ohe.categories_[list(ohe.feature_names_in_).tolist().index(col)][0]
                if isinstance(first_cat, (int, float, np.generic)):
                    input_df[col] = 0
                else:
                    input_df[col] = "unknown"
            except Exception:
                input_df[col] = "unknown"

    
    input_df = input_df[ohe.feature_names_in_]


    input_encoded = ohe.transform(input_df)
    input_dense = input_encoded.toarray()

    
    prediction_proba = model.predict(input_dense)[0][0]
    prediction = (prediction_proba > 0.5).astype(int)
    predicted_class = le.inverse_transform([prediction])[0]

   
    st.subheader("🔮 Prediction Result:")
    
    if predicted_class == "p":
        st.error("⚠️ This mushroom is predicted to be **Poisonous**!")
    else:
        st.success("✅ This mushroom is predicted to be **Edible**.")



