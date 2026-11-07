import os

MODEL_SAVE_PATH = "emotion_model_optimized.keras"
BEST_MODEL_PATH = "best_emotion_model.keras"

print("Current Working Directory:")
print(os.getcwd())

print("\nFinal Model:")
print(os.path.abspath(MODEL_SAVE_PATH))
print("Exists:", os.path.exists(MODEL_SAVE_PATH))

print("\nBest Model:")
print(os.path.abspath(BEST_MODEL_PATH))
print("Exists:", os.path.exists(BEST_MODEL_PATH))