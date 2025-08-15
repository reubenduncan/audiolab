---
title: Audiolab
emoji: 🏃
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 5.42.0
app_file: main.py
pinned: false
---

# AudioLab

Gradio app combining multiple audio generation models for generating speech, music, sound effects, and virtual instrument generators, and more.

## Running

HMR mode for development:

```bash
gradio main.py --demo-name app
```

## Installation

Recommend using a conda environment with python 3.10. 

```bash
conda create -n audiolab python=3.10
conda activate audiolab
pip install torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install xformers separately using the same pytorch version.

```bash
pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
```

Or build from source.

```bash
pip install ninja
pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

## Text-to-Speech (TTS)

Developed on top of Orpheus; the pretrained model is finetuned on top of Llama 3.2 3B to generate both text and encoded audio tokens. Audio encoding is done using SNAC (Multi-Scale Neural Audio Codec).

Support for multiple models in the works:

- Orpheus
- XTTS
- Sesame
- Spark
- Llasa
- Oute

### Voice Cloning

Optimal voice cloning can be achieved with as little as 60 seconds of audio. The workflow uses unsloth to finetune a LoRA on top of the finetuned model. Can be done with zero-shot cloning on top of the finetuned model, but results vary significantly.

### To Do

- Support for multiple models
- Real-time voice streaming

### Resources

https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning


## Music

Music is developed on top of Meta's MusicGen model. MusicGen checkpoints use Google's t5-base and EnCodec 32kHz.