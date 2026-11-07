import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
    CSVLogger
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIG
# ============================================================

# YOUR LOCAL DATASET - DO NOT CHANGE
DATASET_DIR = r"C:\Users\mayan\Desktop\Artificial_Intelligence_DS_ML\Deep_Learning\archive\train"

IMG_SIZE = (100, 100)
BATCH_SIZE = 300
EPOCHS = 40

MODEL_SAVE_PATH = "emotion_model2_optimized.keras"
BEST_MODEL_PATH = "best_emotion2_model.keras"

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

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# TRAINING PARAMETERS
# ============================================================

# Label smoothing inspired by the PyTorch code
LABEL_SMOOTHING = 0.10

# Mixup inspired by the PyTorch code
MIXUP_ALPHA = 0.20

# Initial learning rate
INITIAL_LR = 3e-4

# Minimum learning rate
MIN_LR = 1e-6


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# GPU CHECK
# ============================================================

print("\n==============================================")
print("SYSTEM INFORMATION")
print("==============================================")

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:")
    for gpu in gpus:
        print(gpu)
else:
    print("GPU not detected. Training will use CPU.")


# ============================================================
# DATASET PATH CHECK
# ============================================================

if not os.path.exists(DATASET_DIR):

    raise FileNotFoundError(
        "\nDataset path not found:\n"
        f"{DATASET_DIR}\n\n"
        "Please check DATASET_DIR."
    )

print("\nDataset path:")
print(DATASET_DIR)


# ============================================================
# LOAD TRAIN DATASET
# ============================================================

train_ds = image_dataset_from_directory(

    DATASET_DIR,

    validation_split=0.20,

    subset="training",

    seed=SEED,

    # Explicit class ordering
    class_names=CLASS_NAMES,

    color_mode="grayscale",

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="int",

    shuffle=False
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

val_ds = image_dataset_from_directory(

    DATASET_DIR,

    validation_split=0.20,

    subset="validation",

    seed=SEED,

    # EXACT same class order
    class_names=CLASS_NAMES,

    color_mode="grayscale",

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="int",

    shuffle=False
)


# ============================================================
# DATASET CHECK
# ============================================================

print("\n==============================================")
print("DATASET INFORMATION")
print("==============================================")

print("Detected classes:")
print(train_ds.class_names)

print("\nExpected classes:")
print(CLASS_NAMES)


if train_ds.class_names != CLASS_NAMES:

    raise ValueError(
        "\nClass order mismatch!\n"
        f"Detected: {train_ds.class_names}\n"
        f"Expected: {CLASS_NAMES}"
    )

print("\nClass order: CORRECT")


# ============================================================
# DATASET LABEL DISTRIBUTION
# ============================================================

train_labels = []

for _, labels in train_ds:
    train_labels.extend(labels.numpy())

train_labels = np.array(train_labels)


val_labels = []

for _, labels in val_ds:
    val_labels.extend(labels.numpy())

val_labels = np.array(val_labels)


print("\n==============================================")
print("TRAINING DISTRIBUTION")
print("==============================================")

train_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<10} : {train_counts[i]}"
    )


print("\n==============================================")
print("VALIDATION DISTRIBUTION")
print("==============================================")

val_counts = np.bincount(
    val_labels,
    minlength=NUM_CLASSES
)

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<10} : {val_counts[i]}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(

    [

        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.05
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        tf.keras.layers.RandomContrast(
            0.10
        )

    ],

    name="data_augmentation"
)


# ============================================================
# PREPROCESS TRAIN
# ============================================================

def preprocess_train(images, labels):

    # uint8 -> float32
    images = tf.cast(
        images,
        tf.float32
    )

    # Normalize
    images = images / 255.0

    # Augmentation
    images = data_augmentation(
        images,
        training=True
    )

    # Convert integer labels to one-hot
    labels = tf.one_hot(
        labels,
        depth=NUM_CLASSES
    )

    return images, labels


# ============================================================
# PREPROCESS VALIDATION
# ============================================================

def preprocess_validation(images, labels):

    # uint8 -> float32
    images = tf.cast(
        images,
        tf.float32
    )

    # Normalize only
    images = images / 255.0

    # NO augmentation
    # NO mixup

    labels = tf.one_hot(
        labels,
        depth=NUM_CLASSES
    )

    return images, labels


# ============================================================
# MIXUP
# ============================================================

def mixup_batch(images, labels):

    batch_size = tf.shape(images)[0]

    # Random permutation
    indices = tf.random.shuffle(
        tf.range(batch_size)
    )

    shuffled_images = tf.gather(
        images,
        indices
    )

    shuffled_labels = tf.gather(
        labels,
        indices
    )

    # Sample lambda
    gamma1 = tf.random.gamma(
        shape=[1],
        alpha=MIXUP_ALPHA
    )

    gamma2 = tf.random.gamma(
        shape=[1],
        alpha=MIXUP_ALPHA
    )

    lam = gamma1 / (
        gamma1 + gamma2
    )

    lam = tf.cast(
        lam,
        tf.float32
    )

    # Make lambda broadcastable
    lam_image = tf.reshape(
        lam,
        [1, 1, 1, 1]
    )

    lam_label = tf.reshape(
        lam,
        [1, 1]
    )

    # Mix images
    mixed_images = (
        lam_image * images
        +
        (1.0 - lam_image) * shuffled_images
    )

    # Mix labels
    mixed_labels = (
        lam_label * labels
        +
        (1.0 - lam_label) * shuffled_labels
    )

    return mixed_images, mixed_labels


# ============================================================
# BUILD TRAIN PIPELINE
# ============================================================

train_ds = (
    train_ds
    .cache()
    .shuffle(
        2000,
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .map(
        preprocess_train,
        num_parallel_calls=AUTOTUNE
    )
    .map(
        mixup_batch,
        num_parallel_calls=AUTOTUNE
    )
    .prefetch(AUTOTUNE)
)


# ============================================================
# BUILD VALIDATION PIPELINE
# ============================================================

val_ds = (
    val_ds
    .cache()
    .map(
        preprocess_validation,
        num_parallel_calls=AUTOTUNE
    )
    .prefetch(AUTOTUNE)
)


# ============================================================
# CNN MODEL
# ============================================================

def optimized_cnn(
    input_shape=(100, 100, 1),
    num_classes=7
):

    inputs = Input(
        shape=input_shape,
        name="input_image"
    )


    # ========================================================
    # BLOCK 1
    # ========================================================

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

    x = MaxPooling2D(
        (2, 2)
    )(x)

    x = Dropout(
        0.15
    )(x)


    # ========================================================
    # BLOCK 2
    # ========================================================

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

    x = MaxPooling2D(
        (2, 2)
    )(x)

    x = Dropout(
        0.20
    )(x)


    # ========================================================
    # BLOCK 3
    # ========================================================

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

    x = MaxPooling2D(
        (2, 2)
    )(x)

    x = Dropout(
        0.25
    )(x)


    # ========================================================
    # BLOCK 4
    # ========================================================

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

    x = MaxPooling2D(
        (2, 2)
    )(x)

    x = Dropout(
        0.30
    )(x)


    # ========================================================
    # CLASSIFIER
    # ========================================================

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        256,
        activation="relu",
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)

    x = Dropout(
        0.40
    )(x)

    outputs = Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
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
    learning_rate=INITIAL_LR,
    clipnorm=1.0
)


# ============================================================
# LOSS - LABEL SMOOTHING
# ============================================================

loss_function = tf.keras.losses.CategoricalCrossentropy(
    label_smoothing=LABEL_SMOOTHING
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=optimizer,

    loss=loss_function,

    metrics=[
        tf.keras.metrics.CategoricalAccuracy(
            name="accuracy"
        )
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n==============================================")
print("MODEL SUMMARY")
print("==============================================")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    ModelCheckpoint(

        BEST_MODEL_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    ),


    # --------------------------------------------------------
    # REDUCE LEARNING RATE
    # --------------------------------------------------------

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_delta=0.001,

        min_lr=MIN_LR,

        verbose=1
    ),


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=8,

        min_delta=0.001,

        restore_best_weights=True,

        verbose=1
    ),


    # --------------------------------------------------------
    # SAVE TRAINING LOG
    # --------------------------------------------------------

    CSVLogger(
        "training_history.csv",
        append=False
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n==============================================")
print("TRAINING STARTED")
print("==============================================")

print(f"Initial Learning Rate : {INITIAL_LR}")
print(f"Label Smoothing       : {LABEL_SMOOTHING}")
print(f"Mixup Alpha           : {MIXUP_ALPHA}")
print(f"Epochs                : {EPOCHS}")

print("\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n==============================================")
print("FINAL VALIDATION")
print("==============================================")

val_loss, val_accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print("\n----------------------------------------------")
print(
    f"Validation Loss     : {val_loss:.4f}"
)
print(
    f"Validation Accuracy : {val_accuracy:.4%}"
)
print("----------------------------------------------")


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_SAVE_PATH
)

print("\n==============================================")
print("MODEL SAVED")
print("==============================================")

print(
    f"Final Model : {MODEL_SAVE_PATH}"
)

print(
    f"Best Model  : {BEST_MODEL_PATH}"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n==============================================")
print("VALIDATION PREDICTION ANALYSIS")
print("==============================================")


y_true = []
y_pred = []


for images, labels in val_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    true_classes = np.argmax(
        labels.numpy(),
        axis=1
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        true_classes
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==============================================")
print("CLASSIFICATION REPORT")
print("==============================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n==============================================")
print("CONFUSION MATRIX")
print("==============================================")

print(cm)


# ============================================================
# TRUE / PREDICTED DISTRIBUTION
# ============================================================

true_distribution = np.bincount(
    y_true,
    minlength=NUM_CLASSES
)

pred_distribution = np.bincount(
    y_pred,
    minlength=NUM_CLASSES
)


print("\nTrue validation distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<10} : "
        f"{true_distribution[i]}"
    )


print("\nPredicted validation distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<10} : "
        f"{pred_distribution[i]}"
    )


# ============================================================
# TRAINING CURVES
# ============================================================

history_dict = history.history


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history_dict["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history_dict["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "accuracy_curve.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# LOSS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history_dict["loss"],
    label="Training Loss"
)

plt.plot(
    history_dict["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "loss_curve.png",
    dpi=150
)

plt.show()


# ============================================================
# BEST RESULTS
# ============================================================

best_epoch = np.argmax(
    history_dict["val_accuracy"]
) + 1

best_val_accuracy = max(
    history_dict["val_accuracy"]
)

best_val_loss = min(
    history_dict["val_loss"]
)


print("\n==============================================")
print("BEST TRAINING RESULTS")
print("==============================================")

print(
    f"Best Epoch           : {best_epoch}"
)

print(
    f"Best Val Accuracy    : {best_val_accuracy:.4%}"
)

print(
    f"Best Val Loss        : {best_val_loss:.4f}"
)

print(
    f"Final Learning Rate  : "
    f"{float(model.optimizer.learning_rate.numpy()):.8f}"
)

print("==============================================")

print("\nTraining completed successfully.")