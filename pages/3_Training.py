import streamlit as st
from PIL import Image
from io import StringIO

DEBUG = True

def my_func(files):
    print("hello there")
    print(files)
    
    ################
    # do something #
    ################

    print('Done.')

st.title("학습을 위한 이미지 파일을 업로드 해주세요.")


uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption=uploaded_file.name, width=150)

    # Call Train API
    runbtn = st.button('Train')

    if "runbtn_state" not in st.session_state:
        st.session_state.runbtn_state = False

    if runbtn or st.session_state.runbtn_state:
        st.session_state.runbtn_state = True

        if DEBUG:
            st.write(uploaded_files)
            
        my_func(uploaded_files) # Your function