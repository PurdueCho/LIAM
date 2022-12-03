import streamlit as st
from PIL import Image
from io import StringIO

DEBUG = True

def my_func(path, query):
    print("hello there")
    print("Video path: ", path)
    print("User query: ", query)

    ################
    # do something #
    ################

    st.write('결과:')
    images = ['./imgs/smile man.jpeg']
    times = ['01:42:42']
    st.image(images[0], caption=times[0], width=250)


st.title("자신의 영상속 장면을 검색하세요.")

# Load Video

if 'video_path' not in st.session_state:
        st.session_state['video_path'] = None

uploaded_file = st.file_uploader("Choose a video file [.mp4 only]", type=['mp4'])

if uploaded_file is not None:
    # gets the uploaded video file in bytes
    file_details = {'Filename: ': uploaded_file.name,
                    'Filetype: ': uploaded_file.type,
                    'Filesize: ': uploaded_file.size}
    if DEBUG:
        st.write('\n\nUploaded video file')
        st.write('FILE DETAILS: \n', file_details)
        st.write(uploaded_file)

    # displays the video file
    st.video(uploaded_file)

    # Search Input
    query = st.text_input('찾고 싶은 장면을 검색 하세요.', 'Ex) 웃고있는 이광수...')

    if DEBUG:
        st.write('검색 Query: ', query)

    run_btn = st.button('검색')

    # Call Run API
    if "run_btn_state" not in st.session_state:
        st.session_state.run_btn_state = False

    if run_btn or st.session_state.run_btn_state:
        st.session_state.run_btn_state = True
        my_func(uploaded_file, query) # Your function

