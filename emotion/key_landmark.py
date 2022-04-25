from mediapipe.python.solutions.drawing_utils import DrawingSpec
import enum
from copy import deepcopy

_THICKNESS_DOT = 1
_RADIUS = 2

BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
GRAY_COLOR = (128, 128, 128)
CYAN_COLOR = (255, 255, 0)
CORAL_COLOR = (80, 127, 255)
BROWN_COLOR = (96, 164, 244)

FACE_LANDMARK = dict(
    lips=(61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178,
          87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415),
    left_eye=(263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398),
    left_eyebrow=(276, 283, 282, 295, 285, 300, 293, 334, 296, 336),
    right_eye=(33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173),
    right_eyebrow=(46, 53, 52, 65, 55, 70, 63, 105, 66, 107),
    face_oval=(10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
               149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109),
    nose=(1, 2, 98, 327),
    right_cheek=(205,),
    left_check=(425,),
    midway=(168,),
    # jaw=(32, 146, 176, 208, 171, 148, 199, 175, 152, 428, 396, 377, 282, 369, 400),
    jaw=(200, 199, 175),
)


def _others_generator():
    others = []
    ex = []
    for i in FACE_LANDMARK:
        ex += FACE_LANDMARK[i]

    for index in range(468):
        if index not in ex:
            others.append(index)
    return tuple(others)


def get_key_flm(is_all=False):
    key_flm = deepcopy(FACE_LANDMARK)
    if is_all:
        others = _others_generator()
        key_flm['others'] = others
    return key_flm


def get_face_landmark_style():
    """Returns the default hand landmark drawing style.

    Returns:
        A mapping from each hand landmark to the default drawing spec.
    """
    key_flm_style = get_key_flm(is_all=True)
    style = {
        key_flm_style['lips']:
            DrawingSpec(
                color=RED_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
        key_flm_style['left_eye']:
            DrawingSpec(
                color=BLACK_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS + 1),
        key_flm_style['left_eyebrow']:
            DrawingSpec(
                color=BROWN_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
        key_flm_style['right_eye']:
            DrawingSpec(
                color=BLACK_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS + 1),
        key_flm_style['right_eyebrow']:
            DrawingSpec(
                color=BROWN_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
        key_flm_style['face_oval']:
            DrawingSpec(
                color=BLUE_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
        key_flm_style['jaw']:
            DrawingSpec(
                color=CYAN_COLOR, thickness=_THICKNESS_DOT + 1, circle_radius=_RADIUS + 1),
        key_flm_style['nose']:
            DrawingSpec(
                color=CORAL_COLOR, thickness=_THICKNESS_DOT + 1, circle_radius=_RADIUS + 1),
        key_flm_style['right_cheek']:
            DrawingSpec(
                color=RED_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS + 3),
        key_flm_style['left_check']:
            DrawingSpec(
                color=RED_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS + 3),
        key_flm_style['midway']:
            DrawingSpec(
                color=GRAY_COLOR, thickness=_THICKNESS_DOT + 1, circle_radius=_RADIUS + 1),
        key_flm_style['others']:
            DrawingSpec(
                color=GREEN_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    }
    
    face_landmark_style = {}
    for k, v in style.items():
        for landmark in k:
            face_landmark_style[landmark] = v
    return face_landmark_style
