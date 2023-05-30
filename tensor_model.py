import tensorflow as tf
import tensorflow_datasets as tfds
import cv2 as cv
import numpy as np
import os

rps_datset = tfds.image.RockPaperScissors()
(rps_train_ds, rps_test_ds), ds_info = tfds.load('rock_paper_scissors', split=['train', 'test'], as_supervised=True, shuffle_files=True, with_info=True)
assert isinstance(rps_train_ds, tf.data.Dataset)

tfds.show_examples(rps_train_ds, ds_info, rows=10, cols=10)

# # ds = rps_datset.take(1)

# for image, label in rps_datset:  # example is (image, label)
#   print(image.shape, label)

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

def normalize_img(image, label):
  return tf.cast(image, tf.float32)/255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 86
rps_train_ds = rps_train_ds.map(normalize_img, AUTOTUNE)
rps_train_ds = rps_train_ds.cache()
rps_train_ds = rps_train_ds.shuffle(ds_info.splits['train'].num_examples)
rps_train_ds = rps_train_ds.batch(BATCH_SIZE)
rps_train_ds = rps_train_ds.prefetch(AUTOTUNE)

rps_test_ds = rps_test_ds.map(normalize_img, AUTOTUNE)
rps_test_ds = rps_test_ds.batch(BATCH_SIZE*2)
rps_test_ds = rps_test_ds.prefetch(AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(6, 10, 2, input_shape = (300,300,3), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(48, use_bias=True, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(48, activation='relu'),
  tf.keras.layers.Dense(4, activation='softmax')
  ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(rps_train_ds, epochs=5, callbacks=[cp_callback])
model.evaluate(rps_test_ds)

model.save_weights('end_training/end_weight')

# model.load_weights('end_training/end_weight')
# model.summary()

# cap = cv.VideoCapture(0)
 
# # Check if camera opened successfully
# if (cap.isOpened()== False): 
#   print("Error opening video stream or file")
 
# # Read until video is completed
# while(cap.isOpened()):
#   # Capture frame-by-frame
#   ret, frame = cap.read()
#   if ret == True:
#     resized = cv.resize(frame, (300,300), interpolation = cv.INTER_AREA)
#     normalized = np.expand_dims(resized, axis=0) / 255
#     img_tensor = tf.convert_to_tensor(normalized, dtype=tf.float32)
#     # print(img_tensor.shape)
    
#     result = model.predict(img_tensor)
#     print(np.argmax(result[0]))

#     cv.imshow('Frame',frame)
 
#     # Press Q on keyboard to  exit
#     if cv.waitKey(100) & 0xFF == ord('q'):
#       break

#   else: 
#     break

# cap.release()

# cv.destroyAllWindows()
