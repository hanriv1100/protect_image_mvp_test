import streamlit as st
import cv2
import numpy as np
from PIL import Image


st.set_page_config(page_title="Image Processing MVP", layout="wide")
st.title("Image Processing MVP")
st.markdown("When you upload your image, our protective filter will be applied to ensure it is not used as training data for deepfake purposes")
st.markdown("Please, no refresh")
st.markdown(
    """
    <style>
    .stFileUploader label {
        font-size: 20px;
        font-weight: 500;
        color: #1f77b4;
    }
    .stRadio label {
        font-size: 20px;
        font-weight: 500;
        color: #1f77b4;
    }
    .stRadio div {
        display: flex;
        gap: 20px;
    }
    .custom-caption-1 {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
        padding: 0 0 200px 0;
    }
    .custom-caption-2 {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
        padding: 0 0 30px 0;
    }
    .button-container {
        text-align: center;
        margin-top: 30px;
    }
    .stButton button {
        width: 50%;
        font-size: 25px;
        padding: 10px 20px;
        background-color: #FFFFFF;
        font-weight: bold;
        color: black;
        opacity: 0.8;
        border: 3px solid black;
        border-radius: 5px;
        cursor: pointer;
        margin: 0 auto 50px auto;
        display: block;
    }
    .stButton button:hover {
        background-color: #FFFFFF;
        border: 3px solid #FF0080;
        color: #FF0080;
        opacity: 1;
    }
    .survey {
        text-align: center;
        margin-top: 10px
    }
    .survey-1 {
        font-size: 25px;
        text-align: center;
        margin-top: 10px
        font-weight: bold;
    }
    .survey-2 {
        text-align: center;
        margin-top: 10px
        font-weight: bold;
        padding 0 auto 50px auto
    }
    .a-tag {
        color: #FF0080;
        text-decoration: none;
    }
    a:hover {
        color: #FF0080;
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def change_hair_to_blonde(image):
    # Convert to OpenCV format
    image = np.array(image)
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define the range for hair color (dark colors)
    lower_hair = np.array([0, 0, 0])
    upper_hair = np.array([180, 255, 30])

    # Create a mask for hair
    mask = cv2.inRange(hsv, lower_hair, upper_hair)

    # Change hair color to blonde (light yellow)
    hsv[mask > 0] = (30, 255, 200)

    # Convert back to RGB color space
    image_blonde = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image_blonde

def add_noise(image):
    # Convert to OpenCV format
    image_np = np.array(image)
    # Generate random noise
    noise = np.random.normal(0, 25, image_np.shape).astype(np.uint8)
    # Add noise to the image
    noisy_image = cv2.add(image_np, noise)
    return noisy_image


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.write("Processing...")

    # Save the original image as a numpy array
    image_np = np.array(image)

    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, use_column_width=True)
        st.markdown('<div class="custom-caption-1">Upload Image</div>', unsafe_allow_html=True)

    with col2:
        st.image(image, use_column_width=True)
        st.markdown('<div class="custom-caption-1">Processed Image</div>', unsafe_allow_html=True)




    button_clicked = st.button("Put Upper Pictures into Deepfake Model")
    st.markdown('<p class="survey">If you have used this feature or curious about our technical principles, we would appreciate it if you could respond to the survey below.</p>', unsafe_allow_html=True)
    st.markdown('<p class="survey">We will be giving out gift cards through a monthly raffle among those who leave their contact information.</p>', unsafe_allow_html=True)
    st.markdown('<p class="survey-1"><a href="https://docs.google.com/forms/d/e/1FAIpQLSdzRtuvQyp3CQDhlxEag40v2yDM7u9NYpJ2gv5kgwuNbo1gUA/viewform?usp=sf_link" target="_blank" class="a-tag">Click here! Participating in this Survey would help us!!</a></p>', unsafe_allow_html=True)
    st.markdown('<p class="survey-2">Thank you for using our service!!</p>', unsafe_allow_html=True)

    if button_clicked:
        with col1:
            processed_image = change_hair_to_blonde(image)
            st.image(processed_image, use_column_width=True)
            st.markdown('<div class="custom-caption-2">Upload Image Deepfake Output</div>', unsafe_allow_html=True)
        
        with col2:
            deepfake_image = add_noise(image)
            st.image(deepfake_image, use_column_width=True)
            st.markdown('<div class="custom-caption-2">Processed Image Deepfake Output</div>', unsafe_allow_html=True)
            
            
