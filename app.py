import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image, ExifTags

# 가장 먼저 set_page_config() 호출
st.set_page_config(page_title="딥페이크 사전 방지 필터(테스트)", layout="wide")

# 나머지 Streamlit 코드
st.title("딥페이크 사전 방지 필터(테스트)")
st.markdown("")
st.markdown("<span style='font-size: 18px;'>안녕하세요! 저희는 딥페이크로부터 여러분의 사진을 보호하는 솔루션을 개발하고 있습니다.</span>", unsafe_allow_html=True)
st.markdown("<span style='font-size: 18px;'>저희의 목표는 온라인에 게시된 개인의 사진이 악성 딥페이크 영상에 사용되지 않도록 하는 것입니다. 정식 론칭에 앞서, 여러분의 의견을 듣기 위해 간단한 테스트를 진행하고 있습니다.</span>", unsafe_allow_html=True)
st.markdown("<span style='font-size: 18px;'>최근 SNS에 업로드된 이미지가 딥페이크에 악용되는 사례가 매일 보고되고 있습니다. 따라서 해결책을 강구하기 위해, 여러분의 소중한 의견이 필요합니다.</span>", unsafe_allow_html=True)
st.markdown("")
st.markdown("<span style='font-size: 18px;'>진행 과정 1. 사람 이미지를 업로드하면, 사전 방지 필터가 적용된 이미지를 보여드립니다.</span>", unsafe_allow_html=True)
st.markdown("<span style='font-size: 18px;'>진행 과정 2. 딥페이크 실행 버튼을 클릭하면, 딥페이크 결과를 확인할 수 있습니다.</span>", unsafe_allow_html=True)
st.markdown("<span style='font-size: 18px;'>여러분의 참여에 감사드립니다!</span>", unsafe_allow_html=True)
st.markdown("<span style='font-size: 14px;'> *사전 방지 필터란: 사진에 미세한 변화를 주어 딥페이크 모델에 학습되지 못하도록 방해하는 필터.</span>", unsafe_allow_html=True)
st.markdown(
    """
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <style>
    /* 공통 스타일 */
    .stFileUploader label,
    .stRadio label,
    .stRadio div,
    .custom-caption-1,
    .custom-caption-2,
    .custom-caption-3,
    .button-container,
    .stButton button,
    .survey,
    .survey-1,
    .survey-2,
    .a-tag,
    a {
        transition: all 0.3s ease;
    }
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
    .custom-caption-3 {
        font-size: 30px;
        # font-weight: bold;
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
    
    /* 스마트폰 화면 스타일 */
    @media only screen and (max-width: 600px) {
        .stFileUploader label,
        .stRadio label,
        .stButton button,
        .survey-1 {
            font-size: 16px;
        }
        .custom-caption-1,
        .custom-caption-2 {
            font-size: 24px;
            padding: 0 0 20px 0;
        }
        .custom-caption-3 {
            font-size: 20px;
            padding: 0 0 20px 0;
        }
        .stButton button {
            width: 100%;
            font-size: 18px;
        }
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
        st.markdown('<div class="custom-caption-1">업로드한 원본 이미지</div>', unsafe_allow_html=True)

    with col2:
        st.image(image, use_column_width=True)
        st.markdown('<div class="custom-caption-1">필터를 입힌 이미지</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-caption-1">필터를 입힌 이미지는 육안으로 봤을때, 큰 차이가 없음</div>', unsafe_allow_html=True)

    button_clicked = st.button("딥페이크 적용하기")
    
    col3, col4 = st.columns(2)
    
    if button_clicked:
        processed_image = change_hair_to_blonde(image)
        deepfake_image = add_noise(image)
    else:
        processed_image = None
        deepfake_image = None

    if processed_image is not None and deepfake_image is not None:
        with col3:
            st.image(processed_image, use_column_width=True)
            st.markdown('<div class="custom-caption-2">원본 이미지로 딥페이크를 제작한 경우</div>', unsafe_allow_html=True)
            st.markdown('<div class="custom-caption-3">이해를 돕기 위해 사진에 노란색을 입히는 딥페이크 알고리즘 적용. 원본 이미지는 딥페이크 알고리즘의 영향을 받음.</div>', unsafe_allow_html=True)
        
        with col4:
            st.image(deepfake_image, use_column_width=True)
            st.markdown('<div class="custom-caption-2">방지 필터 이미지로 딥페이크를 제작한 경우</div>', unsafe_allow_html=True)
            st.markdown('<div class="custom-caption-3">방지 필터를 입힌 이미지는 딥페이크의 영향을 받지 않고 알아보기 힘든 사진을 출력함으로써 이미지를 보호함.</div>', unsafe_allow_html=True)
        
        st.markdown('<p class="survey"> 유사한 서비스를 사용해 보셨거나, 저희 기술적 원리에 관심이 있으신 분들께선 아래의 간단한 인터뷰에 참여해 주시면 진심으로 감사드리겠습니다.</p>', unsafe_allow_html=True)
        st.markdown('<p class="survey-1"><a href="https://docs.google.com/forms/d/e/1FAIpQLSfmaI4K6JDbLT-xmRAYlVSwosmoWAPEY01onBfHtzeAbksDuA/viewform?usp=sf_link" target="_blank" class="a-tag">인터뷰로 딥페이크에 맞서기!!</a></p>', unsafe_allow_html=True)
        st.markdown('<p class="survey-2">서비스를 이용해 주셔서 감사합니다! 좋은 하루 보내세요!</p>', unsafe_allow_html=True)
