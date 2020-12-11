import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB1, VGG16
from sklearn.model_selection import StratifiedKFold #정답 비율 클래스에 맞춰서 교차검증
import pandas as pd
import glob

training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
    )

training_datagen2 = ImageDataGenerator(
    rescale=1. / 255
    )
#평가받을 때에는 augmentation option 안넣음

train_generator = training_datagen.flow_from_directory("./data4/",
                                                        batch_size = 32,
                                                        target_size=(256, 256),
                                                        class_mode='categorical',
                                                        subset='training')

valid_generator = training_datagen.flow_from_directory("./data4/",
                                                        batch_size = 32,
                                                        target_size=(256, 256),
                                                        class_mode='categorical',
                                                        subset='validation')

en = VGG16(weights="imagenet",include_top=False,pooling="avg")
en.trainable=False 

model = Sequential()
model.add(en)
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])


es = EarlyStopping(patience=5, monitor='val_loss')
mc = ModelCheckpoint("best.h5",save_best_only = True,monitor='val_loss') #최고만 저장
rp = ReduceLROnPlateau(patience=3, factor=0.2)

#학습
model.fit(train_generator, 
          validation_data = valid_generator, 
          epochs=10, 
          callbacks=[es,mc,rp]) 
          #정답값을 따로 넣지 않아도됨, 이미 train_generator 만들 때 y 값 들어가 있음

model.load_weights("best.h5")

# test_generator = training_datagen2.flow_from_dataframe("./data4/",
#                                                         batch_size = 32,
#                                                         target_size=(256, 256),
#                                                         class_mode='categorical',
#                                                         subset='validation')

print("-- Evaluate --")
scores = model.evaluate_generator(valid_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))


# pred_generator = training_datagen2.flow_from_dataframe(x_pred,
#                                                        x_col="path",
#                                                        y_col=None,
#                                                        target_size=(256,256),
#                                                        class_mode=None,
#                                                        shuffle=False)
print("-- Predict --")
result = model.predict_generator(valid_generator, verbose = 1)
np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x*100)})
# print(result.map(lambda x: np.round_(x, 1)* 100))
print(result)