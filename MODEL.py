import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, BatchNormalization, Activation,
    GlobalAveragePooling2D, Dense, Dropout, Input, Add
)
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

# =========================
# CONFIG
# =========================
DATASET_DIR = r"C:\Users\mayan\Desktop\Artificial_Intelligence_DS_ML\Deep_Learning\archive\train"

IMG_SIZE = (100, 100)     # UPDATED IMAGE SIZE
BATCH_SIZE = 6478
EPOCHS = 40
MODEL_SAVE_PATH = "emotion_model_realworld.h5"

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(CLASS_NAMES)

AUTOTUNE = tf.data.AUTOTUNE

# =========================
# LOAD DATA
# =========================

train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    color_mode="grayscale",
    image_size=IMG_SIZE,     # UPDATED
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    color_mode="grayscale",
    image_size=IMG_SIZE,     # UPDATED
    batch_size=BATCH_SIZE
)

# =========================
# PREPROCESSING + AUGMENTATION
# =========================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")

def preprocess(ds, training=False):
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=AUTOTUNE
    )
    if training:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
        ds = ds.shuffle(1000)
    return ds.cache().prefetch(AUTOTUNE)

train_ds = preprocess(train_ds, training=True)
val_ds   = preprocess(val_ds, training=False)

# =========================
# MINI-XCEPTION WITH RESIDUAL BLOCKS
# =========================

def sep_block(x, filters, strides=1):
    residual = Conv2D(filters, (1, 1), strides=strides, padding="same", use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = DepthwiseConv2D((3, 3), strides=strides, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Add()([x, residual])
    x = Activation("relu")(x)
    return x

def mini_xception(input_shape=(100, 100, 1), num_classes=7):   # UPDATED SHAPE
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = sep_block(x, 64, strides=2)
    x = sep_block(x, 128, strides=2)
    x = sep_block(x, 256, strides=2)
    x = sep_block(x, 256, strides=1)

    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = mini_xception(input_shape=(100, 100, 1), num_classes=NUM_CLASSES)
model.summary()

# =========================
# TRAIN MODEL
# =========================

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_realworld_model.h5",
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=7,
        monitor="val_loss",
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(MODEL_SAVE_PATH)
print(f"\nModel saved as {MODEL_SAVE_PATH}")
