import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import streamlit as st
import streamlit.components.v1 as components

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="딥페이크 사전 방지 필터(테스트)", layout="wide")

ga_code = """
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-1234567890"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-PZPBGNENQG');
</script>
"""

# Streamlit에 GA 코드 삽입
components.html(ga_code, height=0)

st.title("딥페이크 사전 방지 필터(테스트)")
st.markdown("1. 이미지를 업로드하면, 사전 방지 필터를 씌운 이미지를 보여줍니다.    2. 하단의 흰 버튼을 누르면, 딥페이크 모델을 통해 생성된 결과를 보여줍니다.")
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

def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

uploaded_file = st.file_uploader("이미지를 업로드하세요...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = correct_image_orientation(image)
    
    st.write("이미지 처리 중...")

    # Save the original image as a numpy array
    image_np = np.array(image)

    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, use_column_width=True)
        st.markdown('<div class="custom-caption-1">업로드한 이미지</div>', unsafe_allow_html=True)

    with col2:
        st.image(image, use_column_width=True)
        st.markdown('<div class="custom-caption-1">필터를 입힌 이미지</div>', unsafe_allow_html=True)

    button_clicked = st.button("상단의 두 사진을 딥페이크 모델에 학습시키기")
    st.markdown('<p class="survey">위 서비스를 사용해 보셨거나, 저희 기술적 원리에 관심이 있으신 분들께선 아래의 간단한 인터뷰에 참여해 주시면 진심으로 감사드리겠습니다.</p>', unsafe_allow_html=True)
    st.markdown('<p class="survey-1"><a href="https://docs.google.com/forms/d/e/1FAIpQLSdzRtuvQyp3CQDhlxEag40v2yDM7u9NYpJ2gv5kgwuNbo1gUA/viewform?usp=sf_link" target="_blank" class="a-tag">여기를 클릭하여 인터뷰에 응해 주신다면 큰 도움이 될 것 같습니다!!</a></p>', unsafe_allow_html=True)
    st.markdown('<p class="survey-2">서비스를 이용해 주셔서 감사합니다! 좋은 하루 보내세요!</p>', unsafe_allow_html=True)

    if button_clicked:
        with col1:
            processed_image = change_hair_to_blonde(image)
            st.image(processed_image, use_column_width=True)
            st.markdown('<div class="custom-caption-2">원본 이미지를 딥페이크 모델에 넣었을 경우</div>', unsafe_allow_html=True)
        
        with col2:
            deepfake_image = add_noise(image)
            st.image(deepfake_image, use_column_width=True)
            st.markdown('<div class="custom-caption-2">사전 방지 필터 이미지를 딥페이크 모델에 넣었을 경우</div>', unsafe_allow_html=True)
