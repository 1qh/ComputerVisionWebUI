import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from subprocess import check_output
from typing import Generator

import cv2
import numpy as np
import streamlit as st
import yolov5
from dacite import from_dict
from PIL import Image
from streamlit import sidebar as sb
from streamlit.delta_generator import DeltaGenerator
from streamlit_drawable_canvas import st_canvas
from supervision import (
    BoxAnnotator,
    Color,
    ColorPalette,
    Detections,
    LineZone,
    LineZoneAnnotator,
    MaskAnnotator,
    Point,
    PolygonZone,
    PolygonZoneAnnotator,
    VideoInfo,
    crop,
    draw_text,
    get_polygon_center,
)
from ultralytics import NAS, RTDETR, SAM, YOLO
from vidgear.gears import VideoGear


def cvt(f: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def maxcam() -> tuple[int, int]:
    reso = (
        check_output(
            "v4l2-ctl -d /dev/video0 --list-formats-ext | grep Size: | tail -1 | awk '{print $NF}'",
            shell=True,
        )
        .decode()
        .split('x')
    )
    width, height = [int(i) for i in reso] if len(reso) == 2 else (640, 480)
    return width, height


def plur(n: int, s: str) -> str:
    return f"{n} {s}{'s'[:n^1]}" if n else ''


def rgb2hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'


def rgb2ycc(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def avg_rgb(f: np.ndarray) -> np.ndarray:
    return cv2.kmeans(
        cvt(f.reshape(-1, 3).astype(np.float32)),
        1,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )[2][0].astype(np.int32)


def filter_by_vals(d: dict, place: DeltaGenerator, text: str) -> list[int | str]:
    a = list(d.values())

    if place.checkbox(text):
        return [
            a.index(i) for i in place.multiselect(' ', a, label_visibility='collapsed')
        ]
    else:
        return list(d.keys())


def filter_by_keys(d: dict, place: DeltaGenerator, text: str) -> list[int | str]:
    a = list(d.keys())

    if place.checkbox(text):
        return list(place.multiselect(' ', a, label_visibility='collapsed'))
    else:
        return a


def exe_button(place: DeltaGenerator, text: str, cmd: str):
    if place.button(text):
        st.code(cmd, language='bash')
        os.system(cmd)


def mycanvas(
    stroke: str,
    width: int,
    height: int,
    mode: str,
    bg: Image.Image,
    key: str,
):
    return st_canvas(
        stroke_width=2,
        fill_color='#ffffff55',
        stroke_color=stroke,
        width=width,
        height=height,
        drawing_mode=mode,
        background_image=bg,
        key=key,
    )


def legacy_generator(stream: VideoGear, model) -> Generator:
    while True:
        f = stream.read()
        yield f, model(f)


def first_frame(path: str) -> Image.Image:
    stream = VideoGear(source=path).start()
    frame = Image.fromarray(cvt(stream.read()))
    stream.stop()
    return frame


@dataclass
class Toggle:
    fps: bool = True
    count: bool = True
    color: bool = False
    box: bool = True
    label: bool = True
    mask: bool = True
    area: bool = True
    trail: bool = True


@dataclass
class Tweak:
    trail_length: int = 24
    mask_opacity: float = 0.5
    thickness: int = 1
    text_scale: float = 0.5
    text_offset: int = 1
    text_padding: int = 2
    text_color: str = '#000000'


@dataclass
class Draw:
    lines: list = field(default_factory=list)
    zones: list = field(default_factory=list)

    def __str__(self) -> str:
        s = ''
        z = len(self.zones)
        if l := len(self.lines):
            s += '\n - ' + plur(l, 'line')
        if z:
            s += '\n - ' + plur(z, 'zone')
        return s

    def __len__(self) -> int:
        return len(self.lines) + len(self.zones)

    @classmethod
    def from_canvas(cls, d: list):
        return cls(
            lines=[
                (
                    (i['left'] + i['x1'], i['top'] + i['y1']),
                    (i['left'] + i['x2'], i['top'] + i['y2']),
                )
                for i in d
                if i['type'] == 'line'
            ],
            zones=[
                [[x[1], x[2]] for x in k]
                for k in [j[:-1] for j in [i['path'] for i in d if i['type'] == 'path']]
            ]
            + [
                [
                    [i['left'], i['top']],
                    [i['left'] + i['width'], i['top']],
                    [i['left'] + i['width'], i['top'] + i['height']],
                    [i['left'], i['top'] + i['height']],
                ]
                for i in d
                if i['type'] == 'rect'
            ],
        )


@dataclass
class ModelInfo:
    path: str = 'yolov8n.pt'
    classes: list[int] = field(default_factory=list)
    ver: str = 'v8'
    task: str = 'detect'
    conf: float = 0.25
    iou: float = 0.5
    tracker: str | None = None


class Model:
    def __init__(
        self,
        info: ModelInfo = ModelInfo(),
    ):
        self.classes = info.classes
        self.conf = info.conf
        self.iou = info.iou
        self.tracker = info.tracker

        path = info.path
        ver = info.ver

        self.legacy = ver == 'v5'

        match ver:
            case 'sam':
                self.model = SAM(path)
                self.names = []
            case 'rtdetr':
                self.model = RTDETR(path)
                self.names = []  # not available
            case 'NAS':
                self.model = NAS(path)
                self.names = self.model.model.names
            case _:
                self.model = YOLO(path) if not self.legacy else yolov5.load(path)
                self.names = self.model.names

        if self.legacy:
            self.model.classes = self.classes
            self.model.conf = self.conf
            self.model.iou = self.iou

        self.info = info

    def __call__(self, source: str | int) -> Generator:
        if self.legacy:
            stream = VideoGear(source=source).start()
            return legacy_generator(stream, self.model)
        return (
            self.model.predict(
                source,
                stream=True,
                classes=self.classes,
                conf=self.conf,
                iou=self.iou,
                half=True,
                retina_masks=True,
            )
            if self.tracker is None
            else self.model.track(
                source,
                stream=True,
                classes=self.classes,
                conf=self.conf,
                iou=self.iou,
                half=True,
                retina_masks=True,
                tracker=f'{self.tracker}.yaml',
                persist=True,
            )
        )

    def from_res(self, res) -> tuple[Detections, np.ndarray]:
        if self.legacy:
            return Detections.from_yolov5(res[1]), res[0]

        if res.boxes is not None:
            det = Detections.from_yolov8(res)
            if res.boxes.id is not None:
                det.tracker_id = res.boxes.id.cpu().numpy().astype(int)
            return det, cvt(res.plot())

        return Detections.empty(), cvt(res.plot())

    def gen(self, source: str | int) -> Generator:
        start = time.time()
        for res in self(source):
            f = res[0] if self.legacy else res.orig_img
            yield f, self.from_res(res), time.time() - start
            start = time.time()

    def from_frame(self, f: np.ndarray) -> tuple[Detections, np.ndarray]:
        if self.legacy:
            return Detections.from_yolov5(self.model(f)), np.zeros((1, 1, 3))

        res = self.model.predict(
            f,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            half=True,
            retina_masks=True,
        )[0]
        if res.boxes is not None:
            return Detections.from_yolov8(res), cvt(res.plot())

        return Detections.empty(), cvt(res.plot())

    def predict_image(self, file: str | bytes | Path):
        f = np.array(Image.open(file))
        if self.legacy:
            det = Detections.from_yolov5(self.model(f))
            f = BoxAnnotator().annotate(
                scene=f,
                detections=det,
                labels=[f'{conf:0.2f} {self.names[cls]}' for _, _, conf, cls, _ in det],
            )
        else:
            f = cvt(self.from_frame(f)[1])
        st.image(f)

    @classmethod
    def ui(cls, track: bool = True):
        ex = sb.expander('Model', expanded=True)
        tracker = None

        match ex.radio(
            ' ',
            ('YOLO', 'RT-DETR', 'SAM'),
            horizontal=True,
            label_visibility='collapsed',
        ):
            case 'YOLO':
                suffix = {
                    'Detect': '',
                    'Segment': '-seg',
                    'Classify': '-cls',
                    'Pose': '-pose',
                }
                custom = ex.checkbox('Custom weight')
                c1, c2 = ex.columns(2)
                c3, c4 = ex.columns(2)

                ver = c1.selectbox(
                    'Version',
                    ('v8', 'NAS', 'v5u', 'v5', 'v3'),
                    label_visibility='collapsed',
                )
                legacy = ver == 'v5'
                is_nas = ver == 'NAS'
                sizes = ('n', 's', 'm', 'l', 'x')
                has_sizes = ver != 'v3'
                has_tasks = ver == 'v8'

                size = (
                    c2.selectbox(
                        'Size',
                        sizes if not is_nas else sizes[1:4],
                        label_visibility='collapsed',
                    )
                    if has_sizes and not custom
                    else ''
                )
                task = (
                    c3.selectbox(
                        'Task',
                        list(suffix.keys()),
                        label_visibility='collapsed',
                    )
                    if has_tasks and not custom
                    else 'detect'
                )
                if custom:
                    path = c2.selectbox(' ', glob('*.pt'), label_visibility='collapsed')
                else:
                    v = ver[:2] if not is_nas else '_nas_'
                    s = size if has_sizes else ''
                    t = suffix[task] if has_tasks else ''
                    u = ver[2] if len(ver) > 2 and ver[2] == 'u' else ''
                    path = f'yolo{v}{s}{t}{u}.pt'

                if legacy:
                    model = yolov5.load(path)
                else:
                    if is_nas:
                        try:
                            model = NAS(path)
                        except FileNotFoundError:
                            st.warning(
                                'You might want to go to https://docs.ultralytics.com/models to download the weights first.'
                            )
                    else:
                        model = YOLO(path)
                        task = model.overrides['task']
                        path = model.ckpt_path

                        if track:
                            tracker = (
                                c4.selectbox(
                                    'Tracker',
                                    ['bytetrack', 'botsort', 'No track'],
                                    label_visibility='collapsed',
                                )
                                if task != 'classify'
                                else None
                            )
                            tracker = tracker if tracker != 'No track' else None

                    if custom:
                        c3.subheader(f'{task.capitalize()}')

            case 'RT-DETR':
                ver = 'rtdetr'
                task = 'detect'
                size = ex.selectbox('Size', ('l', 'x'))
                path = f'{ver}-{size}.pt'
                model = RTDETR(path)
            case 'SAM':
                ver = 'sam'
                task = 'segment'
                size = ex.selectbox('Size', ('b', 'l'))
                path = f'{ver}_{size}.pt'
                model = SAM(path)

        if ver != 'sam':
            classes = filter_by_vals(model.model.names, ex, 'Custom Classes')
            conf = ex.slider('Threshold', max_value=1.0, value=0.25)
            iou = ex.slider('IoU', max_value=1.0, value=0.5)
        else:
            classes = []
            conf = 0.25
            iou = 0.5

        return cls(
            ModelInfo(
                path=path,
                classes=classes,
                ver=ver,
                task=task,
                conf=conf,
                iou=iou,
                tracker=tracker,
            )
        )


class ColorClassifier:
    def __init__(
        self,
        d: dict = {
            'red': [255, 0, 0],
            'orange': [255, 100, 0],
            'yellow': [255, 200, 0],
            'green': [0, 150, 0],
            'blue': [0, 100, 255],
            'purple': [100, 0, 255],
            'black': [0, 0, 0],
            'white': [255, 255, 255],
        },
    ):
        self.d = d
        self.names = list(d.keys())

        if self.names:
            rgb_mat = np.array(list(d.values())).astype(np.uint8)
            self.ycc = rgb2ycc(rgb_mat)
            self.rgb = [tuple(map(int, i)) for i in rgb_mat]

    def closest(self, _rgb: np.ndarray) -> int:
        return np.argmin(
            np.sum(
                (self.ycc - rgb2ycc(_rgb[np.newaxis])) ** 2,
                axis=1,
            )
        )


class Annotator:
    def __init__(
        self,
        model: Model,
        reso: tuple[int, int],
        draw: Draw = Draw(),
        toggle: Toggle = Toggle(),
        tweak: Tweak = Tweak(),
        color_clf: ColorClassifier = ColorClassifier(),
    ):
        self.model = model
        self.reso = reso
        self.draw = draw
        self.toggle = toggle
        self.tweak = tweak
        self.color_clf = color_clf
        self.unneeded = self.model.info.task in ('classify', 'pose')
        self.ls = [
            LineZone(start=Point(i[0][0], i[0][1]), end=Point(i[1][0], i[1][1]))
            for i in self.draw.lines
        ]
        self.zs = [
            PolygonZone(polygon=np.array(p), frame_resolution_wh=reso)
            for p in self.draw.zones
        ]
        self.text_color = Color.from_hex(tweak.text_color)
        self.line = LineZoneAnnotator(
            thickness=tweak.thickness,
            text_color=self.text_color,
            text_scale=tweak.text_scale,
            text_offset=tweak.text_offset,
            text_padding=tweak.text_padding,
        )
        self.box = BoxAnnotator(
            thickness=tweak.thickness,
            text_color=self.text_color,
            text_scale=tweak.text_scale,
            text_padding=tweak.text_padding,
        )
        self.zones = [
            PolygonZoneAnnotator(
                thickness=tweak.thickness,
                text_color=self.text_color,
                text_scale=tweak.text_scale,
                text_padding=tweak.text_padding,
                zone=z,
                color=ColorPalette.default().by_idx(i),
            )
            for i, z in enumerate(self.zs)
        ]
        self.mask = MaskAnnotator()
        self.pallet = ColorPalette.default()
        self.ques = []
        self.trail_colors = []

    def __dict__(self):
        return {
            'model': asdict(self.model.info),
            'draw': asdict(self.draw),
            'toggle': asdict(self.toggle),
            'tweak': asdict(self.tweak),
            'color': self.color_clf.d,
        }

    def dump(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__(), f, indent=2)

    @classmethod
    def load(cls, path: str, reso: tuple[int, int]):
        d = json.load(open(path))
        return cls(
            model=Model(from_dict(ModelInfo, d['model'])),
            reso=reso,
            toggle=from_dict(Toggle, d['toggle']),
            tweak=from_dict(Tweak, d['tweak']),
            draw=from_dict(Draw, d['draw']),
            color_clf=ColorClassifier(d['color']),
        )

    def __call__(
        self,
        f: np.ndarray,
        det: Detections,
        timetaken: float,
    ) -> np.ndarray:
        begin = time.time()

        tg = self.toggle
        tw = self.tweak
        names = self.model.names

        color_names = self.color_clf.names
        rgb_colors = self.color_clf.rgb

        xyxy = det.xyxy.astype(int)
        class_ids = det.class_id
        tracker_ids = det.tracker_id

        centers = (xyxy[:, [0, 1]] + xyxy[:, [2, 3]]) // 2

        maxlen = tw.trail_length

        if tg.trail and not self.unneeded and tracker_ids is not None:
            for center, c, t in zip(centers, class_ids, tracker_ids):
                t -= 1
                if t >= len(self.ques):
                    self.ques += [deque(maxlen=maxlen)] * (t - len(self.ques) + 1)
                    self.trail_colors += [None] * (t - len(self.trail_colors) + 1)
                self.ques[t].appendleft(tuple(center))
                self.trail_colors[t] = self.pallet.by_idx(c)
            for k, (q, color) in enumerate(zip(self.ques, self.trail_colors)):
                if k + 1 not in tracker_ids:
                    continue
                for i in range(len(q) - 1):
                    cv2.line(
                        f,
                        q[i],
                        q[i + 1],
                        color.as_bgr(),
                        max(1, int((1 - (i / maxlen)) * ((tw.thickness + 1) * 2))),
                    )
        if tg.color and len(color_names):
            naive = False

            for center, bb in zip(centers, xyxy):
                x = center[0]
                y = center[1]

                # for shirt color of person
                # w = bb[2] - bb[0]
                # h = bb[3] - bb[1]
                # cropped = f[
                #     bb[1] : bb[3] - int(h * 0.4),
                #     bb[0] + int(w * 0.2) : bb[2] - int(w * 0.2),
                # ]

                cropped = crop(f, bb)
                if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                    continue

                rgb = f[y, x] if naive else avg_rgb(cropped)
                predict = self.color_clf.closest(rgb)
                r, g, b = rgb_colors[predict]
                draw_text(
                    scene=f,
                    text=color_names[predict],
                    text_anchor=Point(x=x, y=y + 20),
                    text_color=Color(255 - r, 255 - g, 255 - b),
                    text_scale=tw.text_scale,
                    text_padding=tw.text_padding,
                    background_color=Color(r, g, b),
                )
        if tg.box:
            f = self.box.annotate(
                scene=f,
                detections=det,
                labels=[
                    f'{conf:0.2f} {names[cl] if len(names) else cl}'
                    + (f' {track_id}' if track_id else '')
                    for _, _, conf, cl, track_id in det
                ],
                skip_label=not tg.label,
            )
        if tg.mask:
            f = self.mask.annotate(scene=f, detections=det, opacity=tw.mask_opacity)
        if tg.area:
            for a, c in zip(det.area.astype(int), centers):
                draw_text(
                    scene=f,
                    text=str(a)[:-1],
                    text_anchor=Point(x=c[0], y=c[1]),
                    text_color=self.text_color,
                    text_scale=tw.text_scale,
                    text_padding=tw.text_padding,
                )
        for l in self.ls:
            l.trigger(det)
            self.line.annotate(frame=f, line_counter=l)

        for z, zone in zip(self.zs, self.zones):
            z.trigger(det)
            f = zone.annotate(f)

        if tg.count and len(names):
            for i, c in enumerate(np.bincount(class_ids)):
                if c:
                    bg = self.pallet.by_idx(i)
                    r, g, b = bg.as_rgb()
                    draw_text(
                        scene=f,
                        text=plur(c, names[i]),
                        text_anchor=Point(
                            x=f.shape[1] - 60,
                            y=12 + int(i * tw.text_scale * 18),
                        ),
                        text_color=Color(255 - r, 255 - g, 255 - b),
                        text_scale=tw.text_scale,
                        text_padding=tw.text_padding,
                        background_color=bg,
                    )
        if tg.fps:
            fps = 1 / (time.time() - begin + timetaken)
            draw_text(
                scene=f,
                text=f'{fps:.1f}',
                text_anchor=Point(x=50, y=20),
                text_color=self.text_color,
                text_scale=tw.text_scale * 2,
                text_padding=tw.text_padding,
            )
        return f

    def gen(self, source: str | int) -> Generator:
        for f, out, timetaken in self.model.gen(source):
            det, fallback = out
            yield self(f, det, timetaken), fallback

    def from_frame(self, f: np.ndarray) -> np.ndarray:
        start = time.time()
        det, fallback = self.model.from_frame(f)
        return self(f, det, time.time() - start), fallback

    def update(self, f: np.ndarray):
        scale = f.shape[0] / self.reso[1]
        self.ls = [
            LineZone(
                start=Point(i[0][0] * scale, i[0][1] * scale),
                end=Point(i[1][0] * scale, i[1][1] * scale),
            )
            for i in self.draw.lines
        ]
        origin_zs = [
            PolygonZone(polygon=np.array(p), frame_resolution_wh=self.reso)
            for p in self.draw.zones
        ]
        self.zs = [
            PolygonZone(
                polygon=(z.polygon * scale).astype(int),
                frame_resolution_wh=(f.shape[1], f.shape[0]),
            )
            for z in origin_zs
        ]
        for i, z in enumerate(self.zs):
            self.zones[i].zone = z
            self.zones[i].center = get_polygon_center(polygon=z.polygon)

    def native(self, source: str | int):
        cmd = f'{Path(__file__).parent}/native.py --source {source}'
        c1, c2 = st.columns([1, 3])
        c2.subheader(f"Native run on {source if source != 0 else 'camera'}")
        match c2.radio(
            ' ',
            ('Realtime inference', 'Save to video'),
            label_visibility='collapsed',
        ):
            case 'Realtime inference':
                exe_button(c1, 'Show with OpenCV', cmd)
            case 'Save to video':
                saveto = c1.text_input(
                    ' ',
                    'result.mp4',
                    label_visibility='collapsed',
                )
                exe_button(c1, 'Save with OpenCV', f'{cmd} --saveto {saveto}')
        if c1.button('Save config to json'):
            self.dump('config.json')

    @classmethod
    def ui(cls, source: str | int):
        model = Model.ui()
        is_track = model.info.tracker is not None
        names = model.names

        if source:
            reso = VideoInfo.from_video_path(source).resolution_wh
            background = first_frame(source)
        else:
            ex = sb.expander('For camera', expanded=True)
            reso = maxcam()
            if ex.checkbox('Custom resolution', value=True):
                c1, c2 = ex.columns(2)
                reso = (
                    c1.number_input('Width', 1, 7680, 640, 1),
                    c2.number_input('Height', 1, 4320, 480, 1),
                )
            background = None
            if ex.checkbox('Annotate from image'):
                if ex.checkbox('Upload'):
                    background = ex.file_uploader(
                        ' ', label_visibility='collapsed', key='u'
                    )
                if ex.checkbox('Shoot'):
                    background = st.camera_input('Shoot')
            if background:
                model.predict_image(background)
                background = Image.open(background).resize(reso)
            ex.write('**Notes:** Track & line counts only work on native run')

        width, height = reso
        task = model.info.task
        if task in ('pose', 'classify'):
            return cls(model, reso)

        c1, c2 = st.columns([1, 4])
        mode = c1.selectbox(
            'Draw',
            ('line', 'rect', 'polygon')
            if model.tracker is not None
            else ('rect', 'polygon'),
            label_visibility='collapsed',
        )
        bg = background if c2.checkbox('Background', value=True) else None
        stroke, key = ('#fff', 'e') if bg is None else ('#000', 'f')
        canvas = mycanvas(stroke, width, height, mode, bg, key)

        draw = Draw()
        if canvas.json_data is not None:
            draw = Draw.from_canvas(canvas.json_data['objects'])
            c2.markdown(draw)
        if canvas.image_data is not None and len(draw) > 0:
            if c1.button('Export canvas image'):
                Image.alpha_composite(
                    bg.convert('RGBA'),
                    Image.fromarray(canvas.image_data),
                ).save('canvas.png')

        ex1 = sb.expander('Toggle')
        ex2 = sb.expander('Colors')
        ex3 = sb.expander('Visual')
        c1, c2 = ex1.columns(2)
        c3, c4 = ex1.columns(2)
        c5, c6 = ex1.columns(2)
        c7, c8 = ex1.columns(2)

        fps = c1.checkbox('FPS', value=True)
        count = c2.checkbox('Count', value=True) if len(names) else False
        box = c3.checkbox('Box', value=True)
        label = c4.checkbox('Label', value=True) if box else False
        area = c5.checkbox('Area', value=True)
        color = c6.checkbox('Color classify')
        trail = c7.checkbox('Trail', value=True) if is_track else False
        mask = c8.checkbox('Mask', value=True) if task == 'segment' else False

        toggle = Toggle(
            fps=fps,
            count=count,
            color=color,
            box=box,
            label=label,
            mask=mask,
            area=area,
            trail=trail,
        )
        clf = ColorClassifier()
        if toggle.color:
            d = clf.d
            names = filter_by_keys(d, ex2, 'Custom')
            if len(names) > 0:
                d = {k: d[k] for k in names}
                clf = ColorClassifier(d)
                for c, rgb in zip(clf.names, clf.rgb):
                    ex2.color_picker(f'{c}', value=rgb2hex(rgb))
        tweak = Tweak(
            trail_length=ex3.slider('Trail length', 1, 100, 24) if is_track else 0,
            mask_opacity=ex3.slider('Opacity', 0.0, 1.0, 0.5) if mask else 0.0,
            thickness=ex3.slider('Thickness', 0, 10, 1),
            text_scale=ex3.slider('Text size', 0.0, 2.0, 0.5),
            text_offset=ex3.slider('Text offset', 0, 10, 1) if len(draw.lines) else 0,
            text_padding=ex3.slider('Text padding', 0, 10, 2),
            text_color=ex3.color_picker('Text color', '#000000'),
        )
        return cls(
            model=model,
            reso=reso,
            toggle=toggle,
            tweak=tweak,
            draw=draw,
            color_clf=clf,
        )
