from .activation import *
from .gradient import *
from .core import _CAM


def get_interpreter(method) -> _CAM:
    interpreter = {
        "score-cam": ScoreCAM,
        "ss-cam": SSCAM,
        "is-cam": ISCAM,
        "grad-cam": GradCAM,
        "grad-cam-pp": GradCAMpp,
        "xgrad-cam": XGradCAM,
        "layer-cam": LayerCAM,
    }
    return interpreter[method]
