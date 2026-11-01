# Car Make Predictor

A CNN that predicts a car's manufacturer from a photo, built with TensorFlow/Keras.

## How it works

1. **Labels** come from the image filename: the text before the first underscore is the car company.
2. **Balancing** caps every class at the size of the smallest one, so no manufacturer dominates training.
3. **Preprocessing** reads each image in grayscale with OpenCV, resizes it to 100x100 and scales it to the 0 to 1 range.
4. **Model search** compares convolution layer counts (1 to 3), filter sizes (32, 64, 128) and dense layer counts.
5. **Final model** uses two 32 filter convolution blocks with batch normalization and max pooling, then a softmax output, trained with RMSprop.

The architecture diagram is in `Model_Diagram/`.

## Results

The first notebook run reached 67.62% validation accuracy on a balanced subset, with 94.43% on training, so it overfit. [`evaluate.py`](evaluate.py) is a stricter test using all 60,945 images across 31 makes and 287 car models, at 64x64 in color:

| Test split | Test images | Accuracy | Balanced accuracy | Macro F1 |
| :--- | ---: | ---: | ---: | ---: |
| Random split | 12,189 | 55.6% | 56.3% | 0.558 |
| **Unseen car models** | **13,315** | **17.1%** | **15.4%** | **0.121** |

**The gap is the finding.** A random split lets near identical photos of one car model sit in both train and test, so much of that score is recognizing specific cars. On models it has never seen, the network does much worse, so it isn't really learning what makes a BMW a BMW. Always guessing the most common make would score 8.3%, so it has learned something, but this isn't a usable make classifier.

The two runs aren't directly comparable, because the first used a balanced subset, a different input size and grayscale. Pretrained backbones and higher resolution are the obvious next steps.

## Run it

`python evaluate.py` reproduces the table above (`pip install torch pillow scikit-learn`). For the original notebook:

Set `Datadir` and `Testing_dir` in `CNN.ipynb` to your image folders (filenames need the `Company_...jpg` pattern), then run the cells. To view the search runs:

```
tensorboard --logdir Logs
```
