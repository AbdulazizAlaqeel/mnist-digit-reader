# MNIST digit reader

A small neural network that reads handwritten digits, with a web app that
takes a photo of a digit and predicts which one it is.

Built with PyTorch and trained on KAUST's Ibex cluster using SLURM.

## Files

- `mnist_train.py`: loads MNIST, trains the network, tests it and saves the weights
- `mnist.slurm`: SLURM job script that runs the training on Ibex
- `app.py`: Gradio web app that prepares a photo and runs the trained model on it
- `mnist_model.pt`: the trained weights

## Model

A fully connected network: 784 inputs, two hidden layers (128 and 64 neurons,
ReLU), 10 outputs. Trained with SGD, learning rate 0.1, batch size 64.

## Results

Test accuracy on the 10,000 MNIST test images, seed 0:

| Hidden layers | Epochs | Test accuracy |
|---|---|---|
| 1 (128) | 3 | 95.64% |
| 2 (128, 64) | 3 | 95.77% |
| 2 (128, 64) | 10 | 97.96% |

Training longer mattered far more than adding a layer.

## Reading real photos

The model was trained on centered white-on-black digits, so raw phone photos
are read badly. `app.py` prepares each photo the way MNIST images were made:
it separates ink from paper, keeps the digit, thickens thin strokes, scales it
to 20 pixels and centers it in a 28x28 frame.

## Run it

```
pip install -r requirements.txt
python mnist_train.py
python app.py
```