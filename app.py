import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
import dill
import os
import io
import imghdr

# Function to load models safely
def safe_load_model(model_path):
    try:
        model = load_model(model_path, custom_objects={
            'Orthogonal': tf.keras.initializers.Orthogonal(),
            'GlorotUniform': tf.keras.initializers.GlorotUniform(),
            'Zeros': tf.keras.initializers.Zeros()
        })
        return model
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {e}")
        return None

# Function to load model from JSON safely
def safe_load_model_from_json(json_path, weights_path):
    try:
        with open(json_path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {json_path} and {weights_path}: {e}")
        return None

# Define the base path to your files
base_path = os.path.dirname(os.path.abspath(__file__))

# Load models
lstm_model = safe_load_model(os.path.join(base_path, 'LSTM_model.h5'))
cnn_model = safe_load_model_from_json(os.path.join(base_path, 'CNN_MODEL.json'), os.path.join(base_path, 'CNN_MODEL_weights.h5'))
meta_model = safe_load_model(os.path.join(base_path, 'Meta_Model.save.h5'))

# Load additional data needed for predictions
try:
    with open(os.path.join(base_path, 'autism_data.pkl'), 'rb') as f:
        encoded_sequences, categorical_cols, max_sequence_length, X_train_cat = dill.load(f)
    st.write(" ")
except Exception as e:
    st.error(f"Failed to load additional data: {e}")

# Prediction functions
def predict_with_lstm(sequence, categorical_features):
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length, padding='post', truncating='post')
    padded_sequence_reshaped = np.expand_dims(padded_sequence, axis=-1)
    categorical_features_int = np.array([categorical_features]).astype(int)
    prediction = lstm_model.predict([padded_sequence_reshaped, categorical_features_int])
    return prediction[0][0]

def predict_with_cnn(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x /= 255.0  # Rescale if necessary
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    prediction = cnn_model.predict(x)
    return prediction[0][0]

# Security function to check file type
def is_valid_image(file):
    valid_image_formats = ["jpeg", "png", "gif", "bmp"]
    file_format = imghdr.what(file)
    return file_format in valid_image_formats

# Define page functions
def home():
    st.title("Home")
    st.header("Welcome to the Early Autism Prediction in Children App")

    # Load the background images
    background_images = ["B1.jpg", "B2.jpg", "B3.jpg", "B4.jpg", "B5.jpg"]
    
    for img in background_images:
        background_image_path = os.path.join(base_path, "img", img)
        if not os.path.isfile(background_image_path):
            st.error("Background image not found!")
            return
        st.image(background_image_path, use_column_width=True)
    
    st.header("About the Application")
    st.write("Welcome to our application developed to assist parents and caregivers in the early diagnosis of autism spectrum disorder (ASD) in children. Our goal is to provide a user-friendly platform that utilizes behavioral observations and image analysis to predict the likelihood of autism in children at an early age.")
    
    st.header("How it Works")
    st.write("Our application combines machine learning algorithms with behavioral assessments and image recognition technology to offer accurate predictions. Parents and caregivers can answer a series of questions related to the child's behavior, as well as upload an image of the child. Based on this information, our models generate predictions indicating whether the child is likely to be autistic or not.")
    
    st.header("What is Autism")
    st.write("Autism, or autism spectrum disorder (ASD), refers to a broad range of conditions characterized by challenges with social skills, repetitive behaviors, speech and nonverbal communication. According to the Centers for Disease Control, autism affects an estimated 1 in 36 children and 1 in 45 adults in the United States today.")
    
    st.header("What are the Causes of Autism")
    st.write("There are many causes of autism. Research suggests that autism spectrum disorder (ASD) develops from a combination of: Genetic influences and Environmental influences, including social determinants. These factors appear to increase the risk of autism and shape the type of autism that a child will develop. However, it’s important to keep in mind that increased risk is not the same as a cause. For example, some gene changes associated with autism can also be found in people who don’t have the disorder. Similarly, not everyone exposed to an environmental risk factor for autism will develop the disorder. In fact, most will not.")
    
    st.header("What are the Symptoms of Autism")
    st.write("The two core autism symptoms are: Challenges with social communication and interaction skills and Restricted and repetitive behaviors")
    st.write("While autism spectrum disorder looks different from person to person, doctors look for these two symptoms when making a diagnosis. They also rate the severity of these symptoms based on the level of daily support the person requires. Severity levels range from level 1 (requiring support) to level 3 (requiring very substantial support). Not all people with ASD present these two core symptoms the same way. Additionally, some people without ASD may exhibit these signs.")
    
    st.header("Importance of Early Detection")
    st.write("Early detection of autism spectrum disorder is crucial for ensuring timely intervention and support for children. By identifying potential signs of autism at an early age, parents and caregivers can take proactive steps to seek professional evaluation and access appropriate resources and therapies for their child's development.")
    
    st.header("Can Autism Be Prevented?")
    st.write("Can autism be prevented? You can’t prevent autism, but you can lower your risk of having a baby with the condition by taking certain steps, including: Live a healthy lifestyle: Make sure you see your healthcare provider regularly, eat a nutritious diet and exercise. Get prenatal care, and take your provider’s recommended vitamins and supplements. Take care with medications: Ask your healthcare provider which medications are safe and which you should stop taking during your pregnancy. Don’t drink: No kind and no amount of alcohol is safe during pregnancy. Keep up with your vaccinations: Get all of your provider’s recommended vaccines, including the German measles (rubella) vaccine, before you get pregnant. This vaccine can prevent rubella-associated autism.")
    
    st.header("Disclaimer")
    st.write("It's important to note that the predictions provided by our models are for informational purposes only and should not replace professional medical advice. We encourage users to consult healthcare professionals for accurate diagnosis and personalized treatment recommendations tailored to their child's unique needs.")
    
    st.header("Get Started")
    st.write("To get started, simply navigate to the 'Predict' page using the sidebar menu. Follow the instructions to answer the behavioral questions and upload an image of your child. Our models will then generate a prediction indicating the likelihood of autism.")
    
    st.header("Contact Us")
    st.write("Have questions or feedback? We'd love to hear from you! Feel free to reach out to us at [contact@autism-prediction.com](mailto:contact@autism-prediction.com) for any inquiries or assistance.")

def predict():
    st.title("Autism Prediction App")

    col1, col2 = st.columns(2)

    A_questions = [
        "Does your child look at you when you call his/her name?",
        "How easy is it for you to get eye contact with your child?",
        "Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)",
        "Does your child point to share interest with you? (e.g. pointing at an interesting sight)",
        "Does your child pretend? (e.g. care for dolls, talk on a toy phone)",
        "Does your child follow where you’re looking?",
        "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)",
        "Would you describe your child’s first words as:",
        "Does your child use simple gestures? (e.g. wave goodbye)",
        "Does your child stare at nothing with no apparent purpose?"
    ]

    options = {"No": 0, "Yes": 1}
    categorical_features = [0] * len(X_train_cat.columns)
    with col1:
        st.header("ANSWER THE FOLLOWING QUESTIONS")
        st.write("Select 'Yes' or 'No' from the dropdowns")
        st.write("i.e Where Yes = Always, Usually or Sometimes and No = Rarely or Never")
        sequence = [options[st.selectbox(f"Q{i+1} : {A_questions[i]}", options.keys())] for i in range(10)]

    with col2:
        
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        st.markdown(
            """
            <div style="background-color: #BB7BCC; padding: 10px;">A child making eye contact</div>
            """, unsafe_allow_html=True
        )
        background_image_path_6 = os.path.join(base_path, 'img', 'B6.jpg')
        st.image(background_image_path_6, use_column_width=True)

        st.markdown(
            """
            <div style="background-color: #BB7BCC; padding: 10px;">A child pointing at an object</div>
            """, unsafe_allow_html=True
        )
        background_image_path_7 = os.path.join(base_path, 'img', 'B7.jpg')
        st.image(background_image_path_7, use_column_width=True)

    consent = st.checkbox("I consent to the privacy policy and the processing of the uploaded image for prediction purposes.")

    if st.button("Predict") and consent:
        if uploaded_file is not None:
            temp_image_path = os.path.join(base_path, "temp_image.jpg")
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                if lstm_model and cnn_model and meta_model:
                    with st.spinner('Predicting...'):
                        lstm_prediction = predict_with_lstm(sequence, categorical_features)
                        cnn_prediction = predict_with_cnn(temp_image_path)
                        ensemble_prediction = meta_model.predict([np.array([lstm_prediction]), np.array([cnn_prediction])])

                    predicted_class = 1 if ensemble_prediction > 0.5 else 0
                    st.success("Prediction: Autistic" if predicted_class == 1 else "Prediction: Non-Autistic")
                    st.write(f"Predicted Class (Ensemble): {predicted_class}")
                else:
                    st.error("One or more models failed to load. Please check the logs for details.")
            finally:
                # Ensure the temporary image file is deleted after use
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        else:
            st.warning("Please upload an image file.")
    elif not consent:
        st.warning("Please consent to the privacy policy before proceeding.")


def contact():
    st.title("Contact")
    st.write("You can reach us at contact@autism-prediction.com")

def info():
    st.title("Information")
    st.write("This application uses machine learning models to predict the likelihood of autism in children based on MCHAT details and images.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Contact", "Info"])

# Render the chosen page
if page == "Home":
    home()
elif page == "Predict":
    predict()
elif page == "Contact":
    contact()
elif page == "Info":
    info()
