import streamlit as st
from mlflow.pyfunc import load_model
from app.utils import preprocess_image, postprocess_prediction

# Load the deployed model
model = load_model("models/model_1/mlflow_model")

def main():
    st.title("Fovea Location Detection Model Deployment")

    # Get user input (e.g., upload an image)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Preprocess the input image
        image = preprocess_image(uploaded_file)

        # Make a prediction using the loaded model
        prediction = model.predict(image)

        # Postprocess and display the prediction
        st.image(postprocess_prediction(prediction), caption="Prediction", use_column_width=True)

if __name__ == "__main__":
    main()