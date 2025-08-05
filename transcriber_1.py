import whisper
import ffmpeg
import time
import sys
from pytubefix import YouTube

# Load the model (you can use tiny, base, small, medium, or large)
model = whisper.load_model("base")

# Check if the user provided a YouTube URL
if len(sys.argv) < 2:
    print("Usage: python main.py <input>")
    sys.exit(1)

url = sys.argv[1]

if url.startswith("https://www.youtube.com/watch?v="):
    # Download the audio
    yt = YouTube(url)

    uid = f"{int(time.time())}"
    path = yt.streams.filter(only_audio=True).first().download(mp3 = True, filename = uid)

    print(f"Downloaded audio to {path}")
else:
    path = url
    uid = f"{int(time.time())}"

# Convert audio to WAV if necessary
input_audio = path
output_audio = uid + '.wav'
ffmpeg.input(input_audio).output(output_audio).run()

print(f"Converted audio to WAV")
print(f"Transcribing audio...")

# Transcribe audio
result = model.transcribe(uid + ".wav")

print(f"Transcription complete - {uid}")

# Get the transcription text
transcription_text = result['text']

# Save transcription to a text file
with open(uid + ".txt", "w") as f:
    f.write(transcription_text)