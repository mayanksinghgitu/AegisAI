import os
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    MaxPooling2D
)
from tensorflow.keras.preprocessing import image_dataset_from_directory


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = r"C:\Users\mayan\Desktop\Artificial_Intelligence_DS_ML\Deep_Learning\archive\train"

IMG_SIZE = (100, 100)
BATCH_SIZE = 64
EPOCHS = 40

MODEL_SAVE_PATH = "emotion_model_optimized.keras"
BEST_MODEL_PATH = "best_emotion_model.keras"

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

NUM_CLASSES = len(CLASS_NAMES)
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42


# ============================================================
# LOAD DATASET
# ============================================================

train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

val_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("Detected classes:", train_ds.class_names)

# Make sure directory class ordering matches expected ordering
if train_ds.class_names != CLASS_NAMES:
    print("\nWARNING:")
    print("Dataset classes :", train_ds.class_names)
    print("Expected classes:", CLASS_NAMES)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08
        ),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_train(x, y):

    # Normalize to [0, 1]
    x = tf.cast(x, tf.float32) / 255.0

    # Augmentation
    x = data_augmentation(x, training=True)

    return x, y


def preprocess_val(x, y):

    # Normalize only
    x = tf.cast(x, tf.float32) / 255.0

    return x, y


train_ds = (
    train_ds
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(2000, seed=SEED, reshuffle_each_iteration=True)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .cache()
    .prefetch(AUTOTUNE)
)


# ============================================================
# CNN MODEL
# ============================================================

def optimized_cnn(
    input_shape=(100, 100, 1),
    num_classes=7
):

    inputs = Input(shape=input_shape)

    # --------------------------------------------------------
    # BLOCK 1
    # --------------------------------------------------------
    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(inputs)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.15)(x)


    # --------------------------------------------------------
    # BLOCK 2
    # --------------------------------------------------------
    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.20)(x)


    # --------------------------------------------------------
    # BLOCK 3
    # --------------------------------------------------------
    x = Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)


    # --------------------------------------------------------
    # BLOCK 4
    # --------------------------------------------------------
    x = Conv2D(
        256,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(
        256,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.30)(x)


    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        256,
        activation="relu",
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Optimized_Emotion_CNN"
    )

    return model


# ============================================================
# BUILD MODEL
# ============================================================

model = optimized_cnn(
    input_shape=(100, 100, 1),
    num_classes=NUM_CLASSES
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=3e-4
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    # Save best model
    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    # Reduce LR when validation stops improving
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    # Stop training when model stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# FINAL EVALUATION
# ============================================================

val_loss, val_accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print("\n====================================")
print(f"Validation Loss     : {val_loss:.4f}")
print(f"Validation Accuracy : {val_accuracy:.4%}")
print("====================================")


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(MODEL_SAVE_PATH)

print(f"\nFinal model saved as: {MODEL_SAVE_PATH}")
print(f"Best model saved as : {BEST_MODEL_PATH}")