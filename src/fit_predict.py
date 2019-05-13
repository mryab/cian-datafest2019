from glob import glob

import neuralnets as nn
import numpy as np
import pandas as pd
from PIL import Image

train_images = glob("../input/train/train/*/*.jpg")
test_images = glob("../input/test/test/*.jpg")

np.random.seed(42)


def center_crop(img, output_size):
    if isinstance(output_size, int):
        th, tw = output_size, output_size
    else:
        th, tw = output_size
    w, h = img.size
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return img.crop((j, i, j + tw, i + th))


x_train = np.empty((len(train_images), 75, 75, 3), dtype="float32")
y_train = np.empty((len(train_images), 1), dtype="float32")
x_test = np.empty((len(test_images), 75, 75, 3), dtype="float32")

for i, path in enumerate(train_images):
    image_class = path.split("/")[-2]
    x_train[i] = np.array(center_crop(Image.open(path).convert("RGB"), 75))
    if image_class == "indoor":
        y_train[i] = 0
    else:
        y_train[i] = 1
test_paths = []
for i, path in enumerate(test_images):
    x_test[i] = np.array(center_crop(Image.open(path).convert("RGB"), 75))
    test_paths.append(int(path.split("/")[-1].replace(".jpg", "")))

mean = x_train.mean(0, keepdims=True)
x_train -= mean
x_test -= mean
std = x_train.std(0, keepdims=True)
x_train /= std
x_test /= std


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_layers = self.sequential(
            nn.layers.Conv2D(
                3,
                32,
                5,
                1,
                activation=nn.act.relu,
                w_initializer=nn.initializers.RandomUniform(-0.02, 0.02),
            ),
            nn.layers.MaxPool2D(2, 2),
            nn.layers.Dropout(0.2),
            nn.layers.Conv2D(
                32,
                32,
                5,
                1,
                activation=nn.act.relu,
                w_initializer=nn.initializers.RandomUniform(-0.02, 0.02),
            ),
            nn.layers.MaxPool2D(2, 2),
            nn.layers.Dropout(0.2),
            nn.layers.Conv2D(
                32,
                16,
                5,
                1,
                activation=nn.act.relu,
                w_initializer=nn.initializers.RandomUniform(-0.02, 0.02),
            ),
            nn.layers.Flatten(),
            nn.layers.Dropout(0.2),
            nn.layers.Dense(
                1936,
                128,
                activation=nn.act.relu,
                w_initializer=nn.initializers.RandomUniform(-0.02, 0.02),
            ),
            nn.layers.Dropout(0.2),
            nn.layers.Dense(
                128,
                1,
                activation=nn.act.sigmoid,
                w_initializer=nn.initializers.RandomUniform(-0.02, 0.02),
            ),
        )

    def forward(self, x):
        return self.seq_layers.forward(x)


BATCH_SIZE = 128

model = CNN()
model.seq_layers.train()
train_loader = nn.DataLoader(x_train, y_train, batch_size=BATCH_SIZE)
opt = nn.optim.Adam(model.params, 0.001)
loss_fn = nn.losses.SigmoidCrossEntropy()
total_loss = 0
for step in range(400):
    bx, by = train_loader.next_batch()
    by_ = model.forward(bx)
    loss = loss_fn(by_, by)
    if step == 0:
        total_loss = loss.data
    else:
        total_loss = total_loss * 0.9 + loss.data * 0.1
    model.backward(loss)
    opt.step()
    if step % 50 == 0:
        acc = nn.metrics.accuracy(by_.data > 0.5, by)
        print("Step: %i | loss: %.3f | acc: %.2f" % (step, total_loss, acc))
model.seq_layers.eval()
test_preds = np.empty((len(x_test), 1))
for i in range(0, len(x_test), BATCH_SIZE):
    test_preds[i : i + BATCH_SIZE] = model.forward(x_test[i : i + BATCH_SIZE]).data
df = pd.DataFrame({"image_number": test_paths, "prob_outdoor": test_preds.flatten()})
df.to_csv("submission.csv", index=False)
