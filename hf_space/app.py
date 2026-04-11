 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
import gradio as gr
import numpy as np
import os
 
SAMPLE_RATE = 22050
CROP_SEC = 6.0
CROP_LEN = int(SAMPLE_RATE * CROP_SEC)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

GENRES = sorted(["blues", "classical", "country", "disco", "hiphop",
                 "jazz", "metal", "pop", "reggae", "rock"])
GENRE2ID = {g: i for i, g in enumerate(GENRES)}
ID2GENRE = {i: g for i, g in enumerate(GENRES)}

DEVICE = torch.device("cpu")  
class PretrainedEfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.efficientnet = models.efficientnet_b0(weights=None)   
        old = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(
            1, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=False)
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)
 
model = PretrainedEfficientNet(num_classes=10)
weights_path = os.path.join(os.path.dirname(__file__), "best_effnet.pth")
state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)
 
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT,
    hop_length=HOP_LENGTH, n_mels=N_MELS)
db_transform = T.AmplitudeToDB()


def preprocess_audio(audio_tuple): 
    sr, waveform_np = audio_tuple
 
    waveform = torch.tensor(waveform_np, dtype=torch.float32)
 
    if waveform.dim() == 2: 
        waveform = waveform.mean(dim=-1)
    waveform = waveform.unsqueeze(0)   
 
    if waveform.abs().max() > 2.0:
        waveform = waveform / 32768.0
 
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    return waveform


def crop_or_pad(waveform, length): 
    if waveform.shape[1] >= length:
        start = (waveform.shape[1] - length) // 2  
        return waveform[:, start:start + length]
    return F.pad(waveform, (0, length - waveform.shape[1]))


def get_tta_crops(waveform, crop_len): 
    crops = []
    total = waveform.shape[1]
    if total <= crop_len:
        padded = F.pad(waveform, (0, crop_len - total))
        return [padded]

    crops.append(waveform[:, :crop_len])                           
    mid = (total - crop_len) // 2
    crops.append(waveform[:, mid:mid + crop_len])                  
    crops.append(waveform[:, -crop_len:])                           
    return crops


def wave_to_mel(wave): 
    mel = mel_transform(wave)
    mel_db = db_transform(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db
 
@torch.no_grad()
def predict_genre(audio): 
    if audio is None:
        return {g: 0.0 for g in GENRES}

    waveform = preprocess_audio(audio)
    crops = get_tta_crops(waveform, CROP_LEN)
 
    avg_probs = torch.zeros(10)
    for crop in crops:
        mel = wave_to_mel(crop).unsqueeze(0).to(DEVICE)  
        logits = model(mel)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        avg_probs += probs
    avg_probs /= len(crops)

    result = {GENRES[i]: float(avg_probs[i]) for i in range(10)}
    return result
 
DESCRIPTION = """
## Messy Mashup — Music Genre Classifier

Upload a music clip or record from your microphone and the AI will 
identify the genre from 10 categories: **Blues, Classical, Country, Disco, 
HipHop, Jazz, Metal, Pop, Reggae, Rock**.

### How it works
- **Model:** EfficientNet-B0 fine-tuned on 10,000+ synthetic mashups
- **Test-Time Augmentation:** 3 crops (start, middle, end) averaged for robustness
- **Training Score:** 0.90 Macro F1

*Built for BSDA2001P: Introduction to DL and GenAI — IIT Madras*
"""

demo = gr.Interface(
    fn=predict_genre,
    inputs=gr.Audio(
        label="Upload or Record Audio",
        type="numpy"
    ),
    outputs=gr.Label(
        num_top_classes=10,
        label="Genre Prediction"
    ),
    title="Messy Mashup Genre Classifier",
    description=DESCRIPTION,
    examples=[
        ["song0002.wav"],
        ["song0003.wav"],
        ["song0009.wav"]
    ]
)

if __name__ == "__main__":
    demo.launch()
