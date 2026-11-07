
import cv2
import numpy as np
import tensorflow as tf
import time
import os
import webbrowser
import threading
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox


# ============================================================
# CONFIG
# ============================================================

# Emotion model
MODEL_PATH = r"C:\Users\mayan\Desktop\Artificial_Intelligence_DS_ML\Deep_Learning\best_emotion_model.keras"

# Haar Cascade
CASCADE_PATH = r"C:\Users\mayan\Desktop\Artificial_Intelligence_DS_ML\Deep_Learning\haarcascade_frontalface_default.xml"

# IMPORTANT:
# This order MUST match the class order used during model training.
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

IMG_SIZE = 100

# Minimum confidence required
CONFIDENCE_THRESHOLD = 0.30

# Treatment suggestion page
TREATMENT_URL = (
    "file:///C:/Users/mayan/Desktop/"
    "Artificial_Intelligence_DS_ML/Deep_Learning/"
    "AI-Based%20Emotion%20Detection%20System%20for%20Mental%20Health%20and%20Patient%20Care/index.html"
)


# ============================================================
# EMOTION COLORS - BGR
# ============================================================

EMOTION_COLORS = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 128, 0),
    "Fear": (128, 0, 128),
    "Happy": (0, 255, 255),
    "Sad": (255, 0, 0),
    "Surprise": (255, 255, 0),
    "Neutral": (200, 200, 200),
    "Unknown": (0, 255, 0)
}


# ============================================================
# GLOBAL VARIABLES
# ============================================================

running = True
cap = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    print("=" * 60)
    print("[INFO] Loading emotion model...")
    print("[INFO] Model:", model_path)

    model = tf.keras.models.load_model(model_path)

    print("[INFO] Model loaded successfully.")
    print("=" * 60)

    return model


# ============================================================
# LOAD FACE CASCADE
# ============================================================

def load_face_cascade(cascade_path):

    if not os.path.exists(cascade_path):
        raise FileNotFoundError(
            f"Haar cascade file not found:\n{cascade_path}"
        )

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise IOError(
            "Error loading Haar cascade.\n"
            "The cascade file may be corrupted."
        )

    print("[INFO] Haar cascade loaded successfully.")

    return face_cascade


# ============================================================
# PREPROCESS FACE
# ============================================================

def preprocess_face(gray_face):

    # Resize to training image size
    face_resized = cv2.resize(
        gray_face,
        (IMG_SIZE, IMG_SIZE)
    )

    # Normalize
    face_resized = (
        face_resized.astype("float32") / 255.0
    )

    # Add channel dimension
    # Shape: (100, 100, 1)
    face_resized = np.expand_dims(
        face_resized,
        axis=-1
    )

    return face_resized


# ============================================================
# GET EMOTION COLOR
# ============================================================

def get_color_for_emotion(label):

    return EMOTION_COLORS.get(
        label,
        EMOTION_COLORS["Unknown"]
    )


# ============================================================
# VIDEO LOOP
# ============================================================

def video_loop(model, face_cascade):

    global running
    global cap

    print("[INFO] Starting webcam...")

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print("[ERROR] Could not open webcam.")

        running = False

        return

    # Webcam resolution
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        960
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    # FPS calculation
    prev_time = time.time()

    fps = 0.0

    window_name = "Facial Emotion Recognition"

    print("[INFO] Webcam started.")
    print("[INFO] Press 'Q' to exit.")
    print("[INFO] Close the OpenCV window to stop.")

    while running:

        ret, frame = cap.read()

        if not ret:

            print(
                "[ERROR] Failed to grab frame."
            )

            break

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        current_time = time.time()

        dt = current_time - prev_time

        if dt > 0:

            fps = (
                0.9 * fps
                + 0.1 * (1.0 / dt)
            )

        prev_time = current_time

        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )

        face_imgs = []
        face_coords = []

        # ----------------------------------------------------
        # Preprocess detected faces
        # ----------------------------------------------------

        for (x, y, w, h) in faces:

            face_gray = gray[
                y:y + h,
                x:x + w
            ]

            processed = preprocess_face(
                face_gray
            )

            face_imgs.append(processed)

            face_coords.append(
                (x, y, w, h)
            )

        # ----------------------------------------------------
        # Batch prediction
        # ----------------------------------------------------

        if len(face_imgs) > 0:

            batch = np.stack(
                face_imgs,
                axis=0
            )

            preds = model.predict(
                batch,
                verbose=0
            )

            # ------------------------------------------------
            # Process each face
            # ------------------------------------------------

            for i, (
                x,
                y,
                w,
                h
            ) in enumerate(face_coords):

                probs = preds[i]

                label_idx = int(
                    np.argmax(probs)
                )

                confidence = float(
                    np.max(probs)
                )

                # ------------------------------------------------
                # Confidence filtering
                # ------------------------------------------------

                if confidence < CONFIDENCE_THRESHOLD:

                    emotion = "Unknown"

                else:

                    emotion = EMOTION_LABELS.get(
                        label_idx,
                        "Unknown"
                    )

                # ------------------------------------------------
                # Color
                # ------------------------------------------------

                color = get_color_for_emotion(
                    emotion
                )

                # ------------------------------------------------
                # Face rectangle
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

                # ------------------------------------------------
                # Emotion text
                # ------------------------------------------------

                text = (
                    f"{emotion} "
                    f"({confidence * 100:.1f}%)"
                )

                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA
                )

        # ----------------------------------------------------
        # FPS display
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Face count
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Show frame
        # ----------------------------------------------------

        cv2.imshow(
            window_name,
            frame
        )

        # ----------------------------------------------------
        # Q key
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            running = False

            break

        # ----------------------------------------------------
        # Window close detection
        # ----------------------------------------------------

        try:

            if cv2.getWindowProperty(
                window_name,
                cv2.WND_PROP_VISIBLE
            ) < 1:

                running = False

                break

        except cv2.error:

            running = False

            break

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    if cap is not None:

        cap.release()

        cap = None

    cv2.destroyAllWindows()

    print("[INFO] Video loop stopped.")


# ============================================================
# OPEN AEGISAI GUI
# ============================================================

def open_aegisai():

    """
    Launch AegisAI_GUI.py located in the
    same folder as this Python file.
    """

    try:

        # Current Python file directory
        current_folder = os.path.dirname(
            os.path.abspath(__file__)
        )

        # AegisAI GUI path
        aegisai_path = os.path.join(
            current_folder,
            "V3_AegisAI_GUI_Adaptive_Multimodal_Updated.py"
        )

        # Check file
        if not os.path.exists(aegisai_path):

            messagebox.showerror(
                "AegisAI Not Found",
                "AegisAI_GUI.py was not found "
                "in the same folder."
            )

            return

        print("=" * 60)
        print("[INFO] Opening AegisAI GUI...")
        print("[INFO] File:", aegisai_path)
        print("=" * 60)

        # Launch using current Python interpreter
        subprocess.Popen(
            [
                sys.executable,
                aegisai_path
            ],
            cwd=current_folder
        )

    except Exception as e:

        messagebox.showerror(
            "AegisAI Error",
            f"Could not open AegisAI GUI.\n\n{e}"
        )


# ============================================================
# EXIT
# ============================================================

def on_exit(root):

    global running
    global cap

    print("[INFO] Closing application...")

    running = False

    if cap is not None:

        cap.release()

        cap = None

    cv2.destroyAllWindows()

    root.destroy()

    print("[INFO] Application closed.")


# ============================================================
# ABOUT
# ============================================================

def on_about():

    message = (
        "Facial Emotion Recognition System\n"
        "---------------------------------\n\n"
        "• Real-time facial emotion detection\n"
        "• Deep Learning based classification\n"
        "• TensorFlow / Keras\n"
        "• OpenCV Computer Vision\n"
        "• Webcam-based inference\n\n"
        "Developer: Mayank Singh\n"
        "Technology: Python, Deep Learning,\n"
        "Computer Vision and AI"
    )

    messagebox.showinfo(
        "About",
        message
    )


# ============================================================
# TREATMENT SUGGESTION
# ============================================================

def on_treatment():

    try:

        webbrowser.open(
            TREATMENT_URL
        )

    except Exception as e:

        messagebox.showerror(
            "Error",
            f"Could not open treatment page.\n\n{e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global running

    # --------------------------------------------------------
    # GPU configuration
    # --------------------------------------------------------

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    if gpus:

        try:

            for gpu in gpus:

                tf.config.experimental.set_memory_growth(
                    gpu,
                    True
                )

            print(
                f"[INFO] GPU detected: {len(gpus)}"
            )

        except Exception as e:

            print(
                "[WARN] Could not set GPU "
                "memory growth:",
                e
            )

    else:

        print(
            "[INFO] No GPU detected. "
            "Using CPU."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model = load_model(
            MODEL_PATH
        )

    except Exception as e:

        messagebox.showerror(
            "Model Error",
            str(e)
        )

        return

    # --------------------------------------------------------
    # Load Haar Cascade
    # --------------------------------------------------------

    try:

        face_cascade = load_face_cascade(
            CASCADE_PATH
        )

    except Exception as e:

        messagebox.showerror(
            "Face Detection Error",
            str(e)
        )

        return

    # --------------------------------------------------------
    # Tkinter GUI
    # --------------------------------------------------------

    root = tk.Tk()

    root.title(
        "Emotion Recognition Control Panel"
    )

    root.geometry(
        "520x330"
    )

    root.resizable(
        False,
        False
    )

    root.configure(
        bg="#222222"
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title_label = tk.Label(
        root,
        text="Facial Emotion Recognition",
        font=("Segoe UI", 18, "bold"),
        fg="white",
        bg="#222222"
    )

    title_label.pack(
        pady=(18, 5)
    )

    # --------------------------------------------------------
    # Subtitle
    # --------------------------------------------------------

    subtitle_label = tk.Label(
        root,
        text="Webcam-based real-time emotion detection",
        font=("Segoe UI", 10),
        fg="#cccccc",
        bg="#222222"
    )

    subtitle_label.pack(
        pady=(0, 15)
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status_label = tk.Label(
        root,
        text="● Emotion Detection Ready",
        font=("Segoe UI", 10, "bold"),
        fg="#00CC88",
        bg="#222222"
    )

    status_label.pack(
        pady=(0, 10)
    )

    # --------------------------------------------------------
    # Button Frame
    # --------------------------------------------------------

    btn_frame = tk.Frame(
        root,
        bg="#222222"
    )

    btn_frame.pack(
        pady=5
    )

    # ========================================================
    # ABOUT BUTTON
    # ========================================================

    about_btn = tk.Button(
        btn_frame,
        text="About",
        font=("Segoe UI", 11),
        width=18,
        command=on_about,
        bg="#444444",
        fg="white",
        activebackground="#555555",
        activeforeground="white",
        relief="groove",
        bd=2,
        cursor="hand2"
    )

    about_btn.grid(
        row=0,
        column=0,
        padx=8,
        pady=6
    )

    # ========================================================
    # TREATMENT BUTTON
    # ========================================================

    treatment_btn = tk.Button(
        btn_frame,
        text="Treatment Suggestion",
        font=("Segoe UI", 11),
        width=20,
        command=on_treatment,
        bg="#0078D7",
        fg="white",
        activebackground="#0063B1",
        activeforeground="white",
        relief="groove",
        bd=2,
        cursor="hand2"
    )

    treatment_btn.grid(
        row=0,
        column=1,
        padx=8,
        pady=6
    )

    # ========================================================
    # AEGISAI BUTTON
    # ========================================================

    aegisai_btn = tk.Button(
        btn_frame,
        text="Open AegisAI",
        font=("Segoe UI", 11, "bold"),
        width=20,
        command=open_aegisai,
        bg="#00AA88",
        fg="white",
        activebackground="#008F72",
        activeforeground="white",
        relief="groove",
        bd=2,
        cursor="hand2"
    )

    aegisai_btn.grid(
        row=1,
        column=0,
        columnspan=2,
        padx=8,
        pady=10
    )

    # ========================================================
    # EXIT BUTTON
    # ========================================================

    exit_btn = tk.Button(
        root,
        text="Exit",
        font=("Segoe UI", 12, "bold"),
        width=22,
        command=lambda: on_exit(root),
        bg="#CC0000",
        fg="white",
        activebackground="#990000",
        activeforeground="white",
        relief="raised",
        bd=3,
        cursor="hand2"
    )

    exit_btn.pack(
        pady=12
    )

    # --------------------------------------------------------
    # Start video thread
    # --------------------------------------------------------

    running = True

    video_thread = threading.Thread(
        target=video_loop,
        args=(
            model,
            face_cascade
        ),
        daemon=True
    )

    video_thread.start()

    # --------------------------------------------------------
    # Tkinter loop
    # --------------------------------------------------------

    root.mainloop()

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    running = False

    if cap is not None:

        cap.release()

        cap = None

    cv2.destroyAllWindows()


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()
