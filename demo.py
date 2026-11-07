import pandas as pd
df = pd.read_csv(r"C:\Users\mayan\Desktop\Artificial_Intelligence_DS_ML\Deep_Learning\AI-Based Emotion Detection System for Mental Health and Patient Care\DATAset.csv")
print(df.head(4))
df = pd.read_csv(r"C:\Users\mayan\Desktop\Artificial_Intelligence_DS_ML\Deep_Learning\AI-Based Emotion Detection System for Mental Health and Patient Care\DATAset.csv")
size_of_dataset=df.shape[0]
print(size_of_dataset)