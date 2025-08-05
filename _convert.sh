echo "Converting wav files in $1";

for i in $1/*.wav; 
do
    dd bs=58 skip=1 if="$i" of="${i%.*}.mp3";
    ffmpeg -i ${i%.*}.mp3 ${i%.*}.wav
done