import os
os.environ.setdefault("GRADIO_TEMP_DIR", f"/ibex/user/{os.environ['USER']}/gradio_tmp")

import numpy as np
import torch
from torch import nn
from PIL import Image
from scipy import ndimage
import gradio as gr

# Rebuild the same model structure that was trained
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
model.load_state_dict(torch.load("mnist_model.pt"))
model.eval()


def prepare(image):
    """Turn any photo of a digit into a 28x28 picture that looks like MNIST."""
    gray = np.array(image.convert("L"), dtype=np.float32)
    if np.median(gray) > 127:  # light background: make the ink bright
        gray = 255 - gray
    ink = gray > (gray.min() + gray.max()) / 2

    # Find the separate blobs of ink and keep the biggest one
    # that does not touch the edge of the photo: that is the digit.
    labels, count = ndimage.label(ink)
    if count == 0:
        return np.zeros((28, 28), dtype=np.float32)
    edge = set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1])
    sizes = ndimage.sum(ink, labels, range(1, count + 1))
    inside = [i for i in range(1, count + 1) if i not in edge]
    best = max(inside or range(1, count + 1), key=lambda i: sizes[i - 1])
    digit = labels == best

    # Cut the digit out
    rows, cols = np.where(digit)
    digit = digit[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    side = max(digit.shape)

    # Thicken the stroke if it is too thin to survive shrinking
    width = 2 * ndimage.distance_transform_edt(digit).max()
    grow = int((0.1 * side - width) / 2)
    if grow > 0:
        digit = ndimage.binary_dilation(np.pad(digit, grow), iterations=grow)
        side = max(digit.shape)

    # Shrink so the longest side is 20 pixels
    small = Image.fromarray((digit * 255).astype(np.uint8))
    new_size = (max(1, round(20 * digit.shape[1] / side)), max(1, round(20 * digit.shape[0] / side)))
    small = np.array(small.resize(new_size, Image.LANCZOS), dtype=np.float32) / 255

    # Place it on a black 28x28 canvas, centered on its balance point
    cy, cx = ndimage.center_of_mass(small)
    top = min(max(round(14 - cy), 0), 28 - small.shape[0])
    left = min(max(round(14 - cx), 0), 28 - small.shape[1])
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[top:top + small.shape[0], left:left + small.shape[1]] = small
    return canvas


def predict(image):
    if image is None:
        return None
    x = torch.tensor(prepare(image)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return {str(d): float(probs[d]) for d in range(10)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Photo of a digit"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Digit reader",
)
demo.launch()