#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!ssh speech202


# In[2]:


import argparse
import json
import os
import numpy as np
import IPython.display as ipd
from tqdm import tqdm
from scipy.io.wavfile import write

import torch
use_gpu = torch.cuda.is_available()

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)

import params
from model import DiffVC

import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path
import torchaudio


def inference(vc_model_path, src_path, tgt_path, output_path):

    # In[4]:


    # loading voice conversion model
    #vc_path = '/home/rtrt505/speechst1/DiffVC/checkpts/vc/vc_vctk_wodyn.pt' # PRETRAINED MODEL
    #vc_path = '/home/rtrt505/speechst1/DiffVC/logs_dec/vc_150.pt' # MY MODEL
    vc_path = vc_model_path

    generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
                    params.layers, params.kernel, params.dropout, params.window_size, 
                    params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim, 
                    params.beta_min, params.beta_max)
    if use_gpu:
        generator = generator.cuda()
        generator.load_state_dict(torch.load(vc_path))
    else:
        generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
    generator.eval()

    #print(f'Number of parameters: {generator.nparams}')


    # In[5]:


    # loading HiFi-GAN vocoder
    hfg_path = '/home/rtrt505/speechst1/DiffVC/checkpts/vocoder/' # HiFi-GAN path

    with open(hfg_path + 'config.json') as f:
        h = AttrDict(json.load(f))

    if use_gpu:
        hifigan_universal = HiFiGAN(h).cuda()
        hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
    else:
        hifigan_universal = HiFiGAN(h)
        hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator',  map_location='cpu')['generator'])

    _ = hifigan_universal.eval()
    hifigan_universal.remove_weight_norm()


    # In[6]:


    # loading speaker encoder
    enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt') # speaker encoder path
    if use_gpu:
        spk_encoder.load_model(enc_model_fpath, device="cuda")
    else:
        spk_encoder.load_model(enc_model_fpath, device="cpu")


    # In[13]:


    # loading source and reference wavs, calculating mel-spectrograms and speaker embeddings
    #src_path = 'example/6415_111615_000012_000005.wav' # path to source utterance
    #tgt_path = 'example/8534_216567_000015_000010.wav' # path to reference utterance

    src_path = src_path
    tgt_path = tgt_path

    mel_source = torch.from_numpy(get_mel(src_path)).float().unsqueeze(0)
    if use_gpu:
        mel_source = mel_source.cuda()
    mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
    if use_gpu:
        mel_source_lengths = mel_source_lengths.cuda()

    mel_target = torch.from_numpy(get_mel(tgt_path)).float().unsqueeze(0)
    if use_gpu:
        mel_target = mel_target.cuda()
    mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
    if use_gpu:
        mel_target_lengths = mel_target_lengths.cuda()

    embed_target = torch.from_numpy(get_embed(tgt_path)).float().unsqueeze(0)
    if use_gpu:
        embed_target = embed_target.cuda()


    # In[14]:


    # performing voice conversion
    mel_encoded, mel_ = generator.forward(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target, 
                                        n_timesteps=30, mode='ml')
    mel_synth_np = mel_.cpu().detach().squeeze().numpy()
    mel_source_np = mel_.cpu().detach().squeeze().numpy()
    mel = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)
    if use_gpu:
        mel = mel.cuda()

    # In[17]:


    # converted speech
    with torch.no_grad():
        audio = hifigan_universal.forward(mel).cpu().squeeze().clamp(-1, 1)
    sr = 22050
    torchaudio.save(output_path, audio, sr)
    print(f"Converted audio saved to: {output_path}")

    # In[3]:


def get_mel(wav_path):
    wav, _ = load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed

def noise_median_smoothing(x, w=5):
    y = np.copy(x)
    x = np.pad(x, w, "edge")
    for i in range(y.shape[0]):
        med = np.median(x[i:i+2*w+1])
        y[i] = min(x[i+w+1], med)
    return y

def mel_spectral_subtraction(mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5):
    mel_len = mel_source.shape[-1]
    energy_min = 100000.0
    i_min = 0
    for i in range(mel_len - silence_window):
        energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i+silence_window]))
        if energy_cur < energy_min:
            i_min = i
            energy_min = energy_cur
    estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min+silence_window]), axis=-1)
    if smoothing_window is not None:
        estimated_noise_energy = noise_median_smoothing(estimated_noise_energy, smoothing_window)
    mel_denoised = np.copy(mel_synth)
    for i in range(mel_len):
        signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
        estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
        mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
    return mel_denoised


parser = argparse.ArgumentParser(description='T')

parser.add_argument('--vc_model_path', type=str,default='/home/rtrt505/speechst1/DiffVC/checkpts/vc/vc_vctk_wodyn.pt')
parser.add_argument('--src_path', type=str, default='/home/rtrt505/speechst1/DiffVC/example/6415_111615_000012_000005.wav')
parser.add_argument('--tgt_path', type=str, default='/home/rtrt505/speechst1/DiffVC/example/8534_216567_000015_000010.wav')
parser.add_argument('--output_path', type=str, default='/home/rtrt505/speechst1/DiffVC/converted')

argv = parser.parse_args()
'''
diffusion_step = 30
argv.vc_model_path = '/home/rtrt505/speechst1/DiffVC/logs_dec_real_cycle_all_tgt2_time6/vc_290_0408.pt'
argv.src_path = '/home/rtrt505/speechst1/DiffVC/VCTK/wavs/p226/p226_005_mic1.wav'
argv.tgt_path = '/home/rtrt505/speechst1/DiffVC/VCTK/wavs/p229/p229_002_mic1.wav'
argv.output_path = f'/home/rtrt505/speechst1/DiffVC/step_test/test_step_{diffusion_step}.wav'

inference(argv.vc_model_path, argv.src_path, argv.tgt_path, argv.output_path, diffusion_step=diffusion_step)
'''
