import os
import sounddevice as sd
import torch
import torchaudio
from TTS.TTS.tts.configs.xtts_config import XttsConfig
from TTS.TTS.tts.models.xtts import Xtts

CONFIG_PATH = "TTS/voice/tts/custum_ja/config.json"

TOKENIZER_PATH = "TTS/voice/tts/XTTS_v2.0_original_model_files/vocab.json"

XTTS_CHECKPOINT = "TTS/voice/tts/custum_ja/best_model.pth"

SPEAKER_REFERENCE = "TTS/voice/tts/custum_ja/voiceFile0.wav"

OUTPUT_WAV_PATH = "TTS/voice/tts/custum_ja/output.wav"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

print("Inference...")
out = model.inference(
"こんにちは、先生。何か用かな？",
"ja",
gpt_cond_latent,
speaker_embedding,
temperature=0.7, # Add custom parameters here
)
#torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)

# Convert to numpy array and normalize to range [-1, 1]
audio_data = torch.tensor(out["wav"]).numpy()
audio_data = audio_data / max(abs(audio_data))  # Normalize the audio

volume_factor = 0.5  # For example, reduce volume to 50%
audio_data = audio_data * volume_factor

# Play the audio
sd.play(audio_data, samplerate=24000)
sd.wait()  # Wait until audio is done playing