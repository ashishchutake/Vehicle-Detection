import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Lambda

from collections import deque
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
#matplotlib inline

from utils import load_weights, process_output, draw_boxes, preprocess

keras.backend.set_image_dim_ordering('th')

model = Sequential()

#normalize data
model.add(Lambda(lambda x: x/128 - 1,input_shape=(3,448,448)))

#layer 1
model.add(Convolution2D(16, 3, 3,border_mode='same',subsample=(1,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

#layer 2
model.add(Convolution2D(32,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

#layer 3
model.add(Convolution2D(64,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

#layer 4
model.add(Convolution2D(128,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

#layer 5
model.add(Convolution2D(256,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

#layer 6
model.add(Convolution2D(512,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

#layer 7
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))

#layer 8
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))

#layer 9
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())

#layer 10
model.add(Dense(256))

#layer 11
model.add(Dense(4096))
model.add(LeakyReLU(alpha=0.1))

#output layer
model.add(Dense(1470))

model.summary()

load_weights(model,'./yolo-tiny.weights')
test_image = cv2.imread('C:/Users/UIX/Desktop/VehicleDetection/test1.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

plt.imshow(test_image[200:600, 400:1200])
test_image.shape

processed, padhw, shavedim, resized = preprocess(test_image, shave=True)
plt.imshow(resized)

prediction = model.predict(np.array([processed]))[0]
boxes = process_output(prediction, padhw=padhw, shaved=True)
len(boxes)

boxes = remove_duplicates(boxes, test_image)
img =  draw_boxes(boxes, test_image)
plt.figure(figsize=[10,10])
plt.imshow(img)


class Pipeline:
    def __init__(self):
        self.boxes = []
        self.dropped = 0
        self.history = deque(maxlen=8)
        self.first_frames = True

    def apply_threshold(self, boxes):
        if len(boxes) == 0 and len(self.history) > 0:
            self.history.popleft()
        else:
            heatmap = np.zeros([720, 1280], np.int)
            for box in boxes:
                heatmap[box.y1:box.y2, box.x1:box.x2] += 1
                self.history.append(heatmap)
        if len(self.history) > 4:
            self.first_frames = False
        else:
            self.first_frames = True

        if not self.first_frames:
            full_heatmap = np.zeros([720, 1280], np.int)
            for preheat in self.history:
                full_heatmap = np.add(full_heatmap, preheat)

            new_boxes = []
            for box in boxes:
                if full_heatmap[int(box.y), int(box.x)] > 2:
                    new_boxes.append(box)
            return new_boxes

        return boxes

pipe = Pipeline()
test1_video_output = 'C:/Users/UIX/Desktop/VehicleDetection/test1_video_output.mp4'
clip1 = VideoFileClip("C:/Users/UIX/Desktop/VehicleDetection/test1_video.mp4")
lane_clip = clip1.fl_image(pipe.pipeline)
lane_clip.write_videofile(test1_video_output, audio=False)
