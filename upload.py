from datasets import Audio
import sys
import os
import pandas as pd
from datasets import Dataset, DatasetDict

data = []

if len(sys.argv) != 2:
    print("Usage: python transcribe_directory.py /path/to/wav/files")
    sys.exit(1)

directory = sys.argv[1]
if not os.path.isdir(directory):
    print(f"Directory does not exist: {directory}")
    sys.exit(1)

for file in os.listdir(directory):
    if file.endswith(".wav"):
        audio_path = os.path.join(directory, file)
        txt_path = audio_path + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                text = f.read().strip()
            data.append({"audio": audio_path, "text": text})

dataset = Dataset.from_list(data)
dataset = dataset.cast_column("audio", Audio())  # links audio correctly

# Save as Parquet and push
dataset.push_to_hub("reubenduncan/tts_bastila")
