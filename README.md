# Car Make Predictor

A CNN that predicts a car's manufacturer from a photo, built with TensorFlow/Keras.

## How it works

1. **Labels** come from the image filename: the text before the first underscore is the car company.
2. **Balancing** caps every class at the size of the smallest one, so no manufacturer dominates training.
3. **Preprocessing** reads each image in grayscale with OpenCV, resizes it to 100x100 and scales it to the 0 to 1 range.
4. **Model search** compares convolution layer counts (1 to 3), filter sizes (32, 64, 128) and dense layer counts, with runs logged to TensorBoard in `Logs/`.
5. **Final model** uses two 32 filter convolution blocks with batch normalization and max pooling, then a softmax output, trained with RMSprop.

The architecture diagram is in `Model_Diagram/`.

## Results

After 5 epochs with a 20% validation split:

| Set | Accuracy |
| --- | --- |
| Train | 94.43% |
| Validation | 67.62% |

The gap between the two shows the model overfits, and validation loss rises across the later epochs. Dropout, data augmentation and color input are the obvious next things to try.

## Run it

Set `Datadir` and `Testing_dir` in `CNN.ipynb` to your image folders (filenames need the `Company_...jpg` pattern), then run the cells. To view the search runs:

```
tensorboard --logdir Logs
```
