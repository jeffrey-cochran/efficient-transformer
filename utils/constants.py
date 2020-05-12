from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from os import getenv

#
# Default Paths
checkpoint_path = ""
input_fc_dir = ""
input_att_dir = ""
input_box_dir = ""
input_label_h5 = ""
IMAGE_ROOT = getenv("IMAGE_ROOT")

#
# Optimization types
NOAM = "NOAM"
REDUCE_LR = "REDUCE_LR"

#
# Clip grad functions and keys
CLIP_VALUE = "clip_grad_value_"
CLIP_GRAD = "clip_grad_norm_"
#
gradient_clipping_functions = {CLIP_VALUE: clip_grad_value_, CLIP_GRAD: clip_grad_norm_}

#
# Bad endings
bad_endings = [
    "a",
    "an",
    "the",
    "in",
    "for",
    "at",
    "of",
    "with",
    "before",
    "after",
    "on",
    "upon",
    "near",
    "to",
    "is",
    "are",
    "am",
]

#
# Sampling methods
GREEDY = "greedy"
BEAM = "beam_search"

#
# Types of structure loss
SOFTMAX = "softmax_margin"
MARGIN_KEYWORD = "margin"

#
# Aesthetic
