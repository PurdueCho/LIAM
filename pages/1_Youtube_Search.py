import streamlit as st
from PIL import Image
DEBUG = True

def my_func(query):
    print("hello there")
    print(query)

    ################
    # do something #
    ################

    st.write('결과:')
    images = ['./imgs/smile man.jpeg']
    times = ['01:42:42']
    st.image(images[0], caption=times[0], width=250)

def get_Youtube(url):
    if url == 'Ex) https://youtu.be/...':
        pass
    else:
        # displays the video file
        st.video(url)

        # Search Input
        search = st.text_input('찾고 싶은 장면을 검색 하세요.', 'Ex) 웃고있는 이광수...')

        if DEBUG:
            st.write('검색 Query: ', search)

        run_btn = st.button('검색')

        # Call Run API
        if "run_btn_state" not in st.session_state:
            st.session_state.run_btn_state = False

        if run_btn or st.session_state.run_btn_state:
            st.session_state.run_btn_state = True
            my_func(url) # Your function

st.title("원하시는 유튜브 영상속 장면을 검색하세요.")

# Get youtube
url_Youtube = st.text_input('YouTube URL', 'Ex) https://youtu.be/...')

search_btn = st.button('Search')

if "search_btn_state" not in st.session_state:
    st.session_state.search_btn_state = False

if search_btn or st.session_state.search_btn_state:
    st.session_state.search_btn_state = True     
    get_Youtube(url_Youtube) # Your function





