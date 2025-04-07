# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied1 warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.



#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

####################################################################################
# Final CycleDiffusion Train Code  2025.04.02
####################################################################################


import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import params
from data_cycle_4speakers_0810 import VCDecBatchCollate, VCTKDecDataset
from model.vc import DiffVC
from model.utils import FastGL
from utils import save_plot, save_audio

import torch.nn.functional as F
import random

n_mels = params.n_mels
sampling_rate = params.sampling_rate
n_fft = params.n_fft
hop_size = params.hop_size

channels = params.channels
filters = params.filters
layers = params.layers
kernel = params.kernel
dropout = params.dropout
heads = params.heads
window_size = params.window_size
enc_dim = params.enc_dim

dec_dim = params.dec_dim
spk_dim = params.spk_dim
use_ref_t = params.use_ref_t
beta_min = params.beta_min
beta_max = params.beta_max

random_seed = params.seed
test_size = params.test_size

date = '250402'

#data_dir = '/home/rtrt505/speechst1/DiffVC/VCTK'
data_dir = '/home/rtrt505/speechst1/DiffVC/VCTK_2F2M'
val_file = 'filelists/valid.txt'
exc_file = 'filelists/exceptions_vctk.txt'

log_dir = f'real_last_cycle_train_dec_4speakers_iii3_cycle6_from_50_{date}'
enc_dir = 'logs_enc'
#epochs = 110
epochs = 300
#batch_size = 32
#batch_size = 16
batch_size = 4
#learning_rate = 1e-4
learning_rate = 3e-5

use_gpu = torch.cuda.is_available()


if __name__ == "__main__":

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    os.makedirs(log_dir, exist_ok=True)

    print('Initializing data loaders...')
    train_set = VCTKDecDataset(data_dir)
    collate_fn = VCDecBatchCollate()
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                              collate_fn=collate_fn, num_workers=0, drop_last=True, shuffle=True)

    print('Initializing and loading models...')
    fgl = FastGL(n_mels, sampling_rate, n_fft, hop_size).cuda()
    model = DiffVC(n_mels, channels, filters, heads, layers, kernel, 
                   dropout, window_size, enc_dim, spk_dim, use_ref_t, 
                   dec_dim, beta_min, beta_max).cuda()
    model.load_encoder(os.path.join(enc_dir, 'enc.pt'))

    epoch_continue = 50

    
    
    vc_path = f'/home/rtrt505/speechst1/CycleDiffusion/real_last_cycle_train_dec_4speakers_original/vc_{epoch_continue}_0823.pt' # path to voice conversion model

    if use_gpu:
        model = model.cuda()
        model.load_state_dict(torch.load(vc_path))
    else:
        model.load_state_dict(torch.load(vc_path, map_location='cpu'))
    

    print('Encoder:')
    print(model.encoder)
    print('Number of parameters = %.2fm\n' % (model.encoder.nparams/1e6))
    print('Decoder:')
    print(model.decoder)
    print('Number of parameters = %.2fm\n' % (model.decoder.nparams/1e6))

    print('Initializing optimizers...')
    optimizer = torch.optim.Adam(params=model.decoder.parameters(), lr=learning_rate)

    print('Start training.')
    torch.backends.cudnn.benchmark = True
    iteration = 0
    for epoch in range(epoch_continue + 1, epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        model.train()
        losses = []
        cyc_losses = []
        total_losses = []
        for batch in tqdm(train_loader, total=len(train_set)//batch_size):
            
            model.zero_grad()

            mel, mel_ref = batch['mel1'].cuda(), batch['mel2'].cuda()
            c, mel_lengths = batch['c'].cuda(), batch['mel_lengths'].cuda()
            
            loss = model.compute_loss(mel, mel_lengths, mel_ref, c)
            
            mel_tgt = batch['mel_tgt'].cuda()
            tgt_mel_lengths = batch['tgt_mel_lengths'].cuda()
            c_tgt = batch['tgt_c'].cuda()
            cyc_loss = 0
            coef_cyc = 1.0
            diffusion_step = 6
            ################################################
            
            # 반복문 사용
            #print(len(mel))
            '''
            random_element = int(iteration % len(mel))
            print(random_element)
            temp_cyc_loss = 0.0

            mel_source = mel[random_element].unsqueeze(0).float().cuda()
            mel_source_lengths = torch.LongTensor([mel_source.shape[-1]]).cuda()
            embed_source = c[random_element].unsqueeze(0).float().cuda()

            mel_target = mel_tgt[random_element].unsqueeze(0).float().cuda()
            mel_target_lengths = torch.LongTensor([mel_target.shape[-1]]).cuda()
            embed_target = c_tgt[random_element].unsqueeze(0).float().cuda()
                
            _, mel_prime = model(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target, n_timesteps=diffusion_step, mode='ml')
            mel_prime_lengths = torch.LongTensor([mel_prime.shape[-1]]).cuda()
                
            _, mel_double_prime = model(mel_prime, mel_prime_lengths, mel_source, mel_source_lengths, embed_source, n_timesteps=diffusion_step, mode='ml')
                        
            temp_cyc_loss = F.l1_loss(mel_double_prime, mel_source) / (n_mels)

            cyc_loss += coef_cyc * temp_cyc_loss
            
            '''
            
            # 배치 사용
            # cycle: mel -> mel_tgt -> mel
            iii = 3


            with torch.no_grad():
                _, mel_prime = model(mel[:iii], mel_lengths[:iii], mel_tgt[:iii], tgt_mel_lengths[:iii], c_tgt[:iii], n_timesteps=diffusion_step, mode='ml')
            _, mel_double_prime = model(mel_prime, tgt_mel_lengths[:iii], mel[:iii], mel_lengths[:iii], c[:iii], n_timesteps=diffusion_step, mode='ml')         
            cyc_loss = coef_cyc * F.l1_loss(mel_double_prime, mel[:iii]) / n_mels



            ################################################
            
            total_loss = loss + cyc_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1)
            optimizer.step()
            
            losses.append(loss.item())
            cyc_losses.append(cyc_loss.item())
            total_losses.append(total_loss.item())
            iteration += 1
            print()
            print(f"Iteration: {iteration} / loss: {loss}, cyc_loss: {cyc_loss}, total_loss:{total_loss}")

        losses = np.asarray(losses)
        cyc_losses = np.asarray(cyc_losses)
        totoal_losses = np.asarray(total_losses)

        msg = 'Epoch %d: loss = %.4f, cyc_loss = %.4f, total_loss = %.4f\n' % (epoch, np.mean(losses), np.mean(cyc_losses), np.mean(total_losses))
        print(msg)
        with open(f'{log_dir}/{log_dir}.log', 'a') as f:
            f.write(msg)
        losses = []


        if epoch % 10 == 0:
            model.eval()
            print('Inference...\n')
            with torch.no_grad():
                print('Saving model...\n')
                ckpt = model.state_dict()
                torch.save(ckpt, f=f"{log_dir}/vc_{epoch}_{date}.pt")

