# Halite Challenge

Based on https://github.com/brianvanleeuwen/Halite-ML-starter-bot

```
python train_bot.py ./replays
```

## Model

For every point we look at 15*15 patch around it and use a small convolutional network for classification.

The model uses 2D Convolutions with border_mode as 'same' and kernel size of 3.

The exact model looks like

```
Conv2D(32)
ParametricReLU
MaxPool(2, 2)
Conv2D(64)
ParametricReLU
MaxPool(2, 2)
Flatten()
Linear(256)
ParametricReLU
Linear(128)
ParametricReLU
Linear(5)
```

No post processing was applied, I believe a small post processing could have improved results by a good amount, but due to limited time I could put in for this, my major concern was learning.

I used adam with cross entropy loss function.

The replays were downloaded using scrapeData.py script.

## Issues

1. The bot timed out on large board sizes at times because of unavailability of GPUs and convolutions being calculated on CPU.
