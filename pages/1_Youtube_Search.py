import cv2
import streamlit as st
from PIL import Image

DEBUG = True

import plotly.express as px
import datetime
import clip
import torch
from googletrans import Translator
import math
from pytube import Search

device = "cuda" if torch.cuda.is_available() else "cpu"
translator = Translator()


class YoutubeSearch():
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.fps = 0
        self.N = 120
        self.url = ''
        self.video_frames = []

    def my_func(self, url, query):
        print("hello there")
        print("Youtube URL: ", url)
        print("User query: ", query)

        ################
        # do something #
        ################
        query_trans = translator.translate(query, dest='en')
        self.search_video(url, query_trans.text)

        # st.write('결과:')
        # images = ['./imgs/smile man.jpeg']
        # times = ['01:42:42']
        # st.image(images[0], caption=times[0], width=250)

    def download_model(self):
        model, preprocess = clip.load("ViT-B/32", device=device)
        torch.save(model.state_dict(), "./models/model_pretrained.pt")

    def get_Youtube(self, url):
        self.url = url
        if len(url) == 0:
            pass
        elif url.startswith('http'):
            self.getResultsFromURL(url)
        else:
            self.getResultsFromQuery(url)

    #
    def getResultsFromURL(self, url):
        # displays the video file
        result = Search(url)
        movie = result.results[0]
        st.video(url)
        streams = movie.streams.filter(adaptive=True, subtype="mp4", resolution="360p", only_video=True)

        if len(streams) == 0:
            raise Exception("No suitable stream found for this YouTube video!")

        # Download the video as video.mp4
        print("Downloading...")
        streams[0].download(filename="video.mp4")
        print("Download completed.")
        # Search Input
        query = st.text_input('찾고 싶은 장면을 검색 하세요.', '')

        if DEBUG:
            st.write('검색 Query: ', query)

        run_btn = st.button('검색')

        # Call Run API
        if "run_btn_state" not in st.session_state:
            st.session_state.run_btn_state = False

        if run_btn or st.session_state.run_btn_state:
            st.session_state.run_btn_state = True
            self.my_func(url, query)  # Your function

    def getResultsFromQuery(self, query):
        result = Search(query)
        movie = result.results[0]
        video_url = movie.watch_url
        st.video(video_url)

        streams = movie.streams.filter(adaptive=True, subtype="mp4", resolution="360p", only_video=True)

        if len(streams) == 0:
            raise Exception("No suitable stream found for this YouTube video!")

        # Download the video as video.mp4
        print("Downloading...")
        streams[0].download(filename="video.mp4")
        print("Download completed.")

        self.my_func(video_url, query)

    def getVideoFeatures(self):
        # The frame images will be stored in video_frames
        video_frames = []

        # Open the video file
        capture = cv2.VideoCapture('video.mp4')
        self.fps = capture.get(cv2.CAP_PROP_FPS)

        current_frame = 0
        while capture.isOpened():
            # Read the current frame
            ret, frame = capture.read()

            # Convert it to a PIL image (required for CLIP) and store it
            if ret == True:
                video_frames.append(Image.fromarray(frame[:, :, ::-1]))
            else:
                break

            # Skip N frames
            current_frame += self.N
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        batch_size = 256
        batches = math.ceil(len(video_frames) / batch_size)
        self.video_frames = video_frames

        # The encoded features will bs stored in video_features
        video_features = torch.empty([0, 512], dtype=torch.float16).to(device)

        # Process each batch
        for i in range(batches):
            print(f"Processing batch {i + 1}/{batches}")

            # Get the relevant frames
            batch_frames = video_frames[i * batch_size: (i + 1) * batch_size]

            # Preprocess the images for the batch
            batch_preprocessed = torch.stack([self.preprocess(frame) for frame in batch_frames]).to(device)

            # Encode with CLIP and normalize
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_preprocessed)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)

            # Append the batch to the list containing all features
            video_features = torch.cat((video_features, batch_features))
        return video_features

    def search_video(self, video_url, search_query, display_heatmap=False, display_results_count=3):
        # Encode and normalize the search query using CLIP
        with torch.no_grad():
            text_features = self.model.encode_text(clip.tokenize(search_query).to(device))
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute the similarity between the search query and each frame using the Cosine similarity
        video_features = self.getVideoFeatures()
        similarities = (100.0 * video_features @ text_features.T)
        values, best_photo_idx = similarities.topk(display_results_count, dim=0)

        # Display the heatmap
        if display_heatmap:
            print("Search query heatmap over the frames of the video:")
            fig = px.imshow(similarities.T.cpu().numpy(), height=50, aspect='auto', color_continuous_scale='viridis')
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            fig.show()
            print()

        # Display the top 3 frames
        for frame_id in best_photo_idx:
            print(frame_id)
            seconds = round(frame_id.cpu().numpy()[0] * self.N / self.fps)
            st.image(self.video_frames[frame_id], caption=f"Found at {str(datetime.timedelta(seconds=seconds))}",
                     width=800)
            url_with_timestamp = f"{video_url}&t={seconds}"
            link = f'check out this [link]({url_with_timestamp})'
            st.markdown(link, unsafe_allow_html=True)
            st.text('\n\n\n')
            # st.text(HTML("Link (<a target=\"_blank\" href=\"{video_url}&t={seconds}\">link</a>)"))
        # display(video_frames[frame_id])
        #
        # # Find the timestamp in the video and display it
        # seconds = round(frame_id.cpu().numpy()[0] * N / fps)
        # display(HTML(f"Found at {str(datetime.timedelta(seconds=seconds))} (<a target=\"_blank\" href=\"{video_url}&t={seconds}\">link</a>)"))


def main() -> None:
    # download_model()
    st.title("원하시는 유튜브 영상속 장면을 검색하세요.")

    # Get youtube
    url_Youtube = st.text_input('검색어 혹은 URL을 입력하세요', '')
    search_btn = st.button('Search')
    if "search_btn_state" not in st.session_state:
        st.session_state.search_btn_state = False

    if search_btn or st.session_state.search_btn_state:
        st.session_state.search_btn_state = True
        youtubeSearch = YoutubeSearch()
        youtubeSearch.get_Youtube(url_Youtube)  # Your function


if __name__ == "__main__":
    main()
