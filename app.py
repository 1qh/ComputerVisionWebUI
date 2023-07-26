import os
from pathlib import Path
from shutil import which
from time import gmtime, strftime

import cv2
import streamlit as st
from av import VideoFrame
from lightning import LightningApp, LightningFlow
from lightning.app.frontend import StreamlitFrontend
from psutil import process_iter
from streamlit import session_state, set_page_config
from streamlit import sidebar as sb
from streamlit_webrtc import webrtc_streamer
from supervision import VideoInfo

from core import Annotator, Model, cvt

_shape = None

if 'path' not in session_state:
    session_state['path'] = ''


def local_css(file: str):
    with open(file) as f:
        st.markdown(
            f'<style>{f.read()}</style>',
            unsafe_allow_html=True,
        )


def st_config():
    set_page_config(
        page_icon='🎥',
        page_title='ComputerVisionWebUI',
        layout='wide',
        initial_sidebar_state='expanded',
        menu_items={
            'Report a bug': 'https://github.com/1qh/ComputerVisionWebUI/issues/new',
        },
    )
    try:
        local_css('style.css')
    except FileNotFoundError:
        st.markdown(
            """
            <style>
            div[data-testid="stExpander"] div[role="button"] p {
              font-size: 1.5rem;
            }
            div.stButton button {
              width: 100%;
              border-radius: 20px;
            }
            div.block-container {
              padding-top: 2rem;
            }
            footer {
              visibility: hidden;
            }
            @font-face {
              font-family: "SF Pro Display";
            }
            html,
            body,
            [class*="css"] {
              font-family: "SF Pro Display";
            }
            thead tr th:first-child {
              display: none;
            }
            tbody th {
              display: none;
            }
            </style>
            """
        )


def hms(s: int) -> str:
    return strftime('%H:%M:%S', gmtime(s))


def trim_vid(path: str, begin: str, end: str) -> str:
    trim = f'trim_{path[3:]}'
    os.system(f'ffmpeg -y -i {path} -ss {begin} -to {end} -c copy {trim}')
    return trim


def prepare(path: str, place):
    vid = VideoInfo.from_video_path(path)

    if which('ffmpeg'):
        if trimmed := place.checkbox('Trim'):
            length = int(vid.total_frames / vid.fps)
            begin, end = place.slider(
                ' ',
                value=(0, length),
                max_value=length,
                label_visibility='collapsed',
            )
            begin, end = hms(begin), hms(end)
            if place.button(f'Trim from {begin[3:]} to {end[3:]}'):
                path = trim_vid(path, begin, end)
                session_state['path'] = path
        else:
            session_state['path'] = path
    else:
        session_state['path'] = path


def main(state):
    st_config()
    file = sb.file_uploader(' ', label_visibility='collapsed')
    running = sb.checkbox('Realtime inference (slower than native)')
    usecam = sb.checkbox('Use camera')
    mt = st.empty()

    if usecam:
        file = None

        an = Annotator.ui(0)
        an.native(0)
        width, height = an.reso

        cap = cv2.VideoCapture(0)
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(6, codec)
        cap.set(5, 30)
        cap.set(3, width)
        cap.set(4, height)

        while running:
            t1, t2 = mt.tabs(['Main', 'Fallback'])
            if an.unneeded:
                t1, t2 = t2, t1
            success, f = cap.read()
            if success:
                f, fallback = an.from_frame(f)
                t1.image(cvt(f))
                t2.image(fallback)
            else:
                break
        cap.release()

        def cam_stream(key, callback):
            webrtc_streamer(
                key=key,
                video_frame_callback=callback,
                media_stream_constraints={
                    'video': {
                        'width': {'min': width},
                        'height': {'min': height},
                    }
                },
            )

        def simplecam(frame):
            f = an.from_frame(frame.to_ndarray(format='bgr24'))[1]
            return VideoFrame.from_ndarray(f)

        # oh my god, it took me so long to realize the frame bigger through time
        def cam(frame):
            f = frame.to_ndarray(format='bgr24')
            global _shape
            if f.shape != _shape:
                _shape = f.shape
                an.update(f)
            f = cvt(an.from_frame(f)[0])
            return VideoFrame.from_ndarray(f)

        if an.unneeded:
            cam_stream('a', simplecam)
        else:
            cam_stream('b', cam)

    if file:
        ex = sb.expander('Uploaded file')
        if 'image' in file.type:
            ex.image(file)
            Model.ui(track=False).predict_image(file)

        elif 'video' in file.type:
            ex.video(file)
            path = f'up_{file.name}'
            with open(path, 'wb') as up:
                up.write(file.read())

            prepare(path, ex)
            path = session_state['path']
            vid = VideoInfo.from_video_path(path)
            reso = vid.resolution_wh
            total_frames = vid.total_frames

            ex.markdown(
                f'''
            - Video resolution: {'x'.join([str(i) for i in reso])}
            - Total frames: {total_frames}
            - FPS: {vid.fps}
            - Path: {path}
                '''
            )
            an = Annotator.ui(path)
            an.native(path)

            count = 0

            while running:
                for f, fallback in an.gen(path):
                    t1, t2 = mt.tabs(['Main', 'Fallback'])
                    if an.unneeded:
                        t1, t2 = t2, t1
                    t1.image(cvt(f))
                    t2.image(fallback)
                    count += 1
                    t1.progress(count / total_frames)

        else:
            sb.warning('Please upload image/video')


class App(LightningFlow):
    def configure_layout(self):
        return StreamlitFrontend(render_fn=main)

    def run(self):
        pass


lit = LightningApp(App())

running_apps = [i for i in [p.cmdline() for p in process_iter()] if 'run' in i]
this_process = next(p for p in running_apps if any(Path(__file__).stem in a for a in p))

if 'app' not in this_process:
    main('')
