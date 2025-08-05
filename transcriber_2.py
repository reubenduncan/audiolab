import whisper
import os
import sys

# Usage: python transcribe_directory.py /path/to/wav/files

if len(sys.argv) != 2:
    print("Usage: python transcribe_directory.py /path/to/wav/files")
    sys.exit(1)

input_dir = sys.argv[1]
if not os.path.isdir(input_dir):
    print(f"Directory does not exist: {input_dir}")
    sys.exit(1)

model = whisper.load_model("base")  # Choose: tiny, base, small, medium, large

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".wav"):
        filepath = os.path.join(input_dir, filename)
        print(f"Transcribing {filename}...")
        result = model.transcribe(filepath)
        out_path = os.path.join(input_dir, filename + ".txt")
        with open(out_path, "w") as f:
            f.write(result["text"])
        print(f"Saved transcript to {out_path}")
