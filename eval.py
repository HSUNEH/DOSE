from inst_decoder import InstDecoderModule, InstDecoderConfig
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import os
import audiofile as af
import pretty_midi
import matplotlib.pyplot as plt
import argparse
import torch
from tqdm import tqdm
import dac
from dac.utils import load_model
from dac.model import DAC
import scipy.io.wavfile
from einops import rearrange
import numpy as np
import os
import shutil
import librosa
from torch.utils.data import Dataset
import random

# For reproducibility.
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

class Getwav(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.wav_file = os.listdir(file_path)
    def __len__(self):
        return len(self.wav_file)
    def __getitem__(self, idx):
        # wav = af.read(self.file_path + self.wav_file[idx])[0]
        wav, sr = librosa.load(self.file_path + self.wav_file[idx], sr=None, mono=False)
        if sr != 44100:
            wav = librosa.resample(wav, sr, 44100)
        padded_wav = np.zeros((2, 44100*4))
        padded_wav[:,:wav.shape[1]] = wav[:,:44100*4] # 앞의 4초만 사용
        padded_wav = torch.tensor(padded_wav).float()
        padded_wav = padded_wav.unsqueeze(0)
        # dac 돌리고 
        if gpu:
            padded_wav = padded_wav.cuda()

        with torch.no_grad(): # dim ["latents" 72, "codes" 9 , "z" 1024] 
            processed = dac_model.preprocess(torch.mean(padded_wav, dim=1, keepdim=True).float(), 44100) # stereo to mono, (batch, 1, seq(44100*4))
            z, codes, latents,_ ,_ = dac_model.encode(processed)
            x = codes.detach().long() # size:(batch, 9, 345)
        audio_rep = x.squeeze(0)
        # delay pattern 만들기
        audio_rep = audio_rep.cpu()
        audio_rep_l = tokenize_inst(audio_rep,4) # MONO, (9, 345+8)
        audio_dac = rearrange(audio_rep_l, 'd t -> t d') # (345+8, 9)
        return audio_dac, torch.tensor(0).float(), idx # (9, 345+8), 0

def tokenize_inst(inst_dac_l,length): # inst_dac_l : b, 9, seq (345 or 173)    

    if length == 4: 
        all_tokens_np = np.zeros(((inst_dac_l.shape[0]),345+8), dtype=np.int32)  
    elif length ==2 :
        all_tokens_np = np.zeros((inst_dac_l.shape[0],1+173+8+1), dtype=np.int32)  
    elif length ==0 :
        all_tokens_np = np.zeros((inst_dac_l.shape[0],1+inst_dac_l.shape[1]+8+1), dtype=np.int32)  
    # interleaving pattern
    for i, codes in enumerate(inst_dac_l): # dac : (9, (max)345)
        if length == 4:
            start = i
        else: 
            start = i+1
        end = start + inst_dac_l.shape[1]

        all_tokens_np[i, start: end] = codes.numpy() + 1 # +2000
    return all_tokens_np # (9,345)

def dataset_wav_load(data_dir,BATCH_SIZE):
    dataset = Getwav(data_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    # mixed_loop(2, 176400), kick(2, 88200), snare(2, 88200), hhclosed(2, 88200)
    return dataloader
    
def audio_padding(audio, length=44100):
    if audio.shape[1] < length: # ex) audio.shape = (1, 18432)
        padding = np.zeros((1, int(length) - audio.shape[1]))
        audio = np.concatenate((audio, padding), axis=1)
    elif audio.shape[1] > length:
        audio = audio[:,:int(length)]
    return audio

def inst_generate(test_dataloader, inst, args):
    idx = 1
    error_list = []
    for batch_idx, batch in (enumerate(test_dataloader)):
        # mixed_loops, inst_shot = batch
        mixed_loops, inst_shot, num_idx = batch
        num_idx = num_idx.item()
        if gpu:
            mixed_loops = mixed_loops.cuda()
            inst_shot = inst_shot.cuda()
        batch = mixed_loops, inst_shot
        
        end = False
        # y_pred, end, loss, padding_losses, padding_in_losses, audio_losses = model(batch) # torch.Size([1, seq_len, 9]), False
        y_pred, end = model(batch) # torch.Size([1, seq_len, 9]), False

        if dac_only:
            if not torch.all(y_pred[:,353,:] == 0):
                print('Start Token FUCKED')
            if not torch.all(y_pred[:,-1,:] == 0):
                print('End Token FUCKED')

            inst_tokens = y_pred[:,354:-1,:]
            # Audio part only -> inst_tokens
        else: 
            if not torch.all(y_pred[:,-1,:] == 0):
                print('End Token FUCKED')
            inst_tokens = y_pred[:,1:-1,:]

        ### 푸는 과정

        inst_tokens = inst_tokens - 1
        mixed_loops = mixed_loops -1
        
        
        dac_len = inst_tokens.shape[1]-8
        mixed_len = mixed_loops.shape[1]-8
        
        
        # interleaving pattern 풀기
        for i in range(1,9): # dac : (9, (max)431) / codes : (s, d)
            inst_tokens[:,:dac_len,i] = inst_tokens[:,i:i+dac_len,i] 
            mixed_loops[:,:mixed_len,i] = mixed_loops[:,i:i+mixed_len,i]
            
            
        inst_tokens = inst_tokens[:,:-8,:]
        mixed_loops = mixed_loops[:,:-8,:]

        if inst_tokens.min() < 0 or inst_tokens.max() > 1023:
            error_list.append(idx)
        else:
            for _, codes in enumerate([inst_tokens]): # inst_tokens : [1,40,9]
                inst_codes = codes.permute(0,2,1) # inst_codes : torch.Size([1, 9, 40])
                latent = dac_model.quantizer.from_codes(inst_codes)[0] # latent : [1, 1024, 40]
                audio = dac_model.decode(latent)[0] # torch.Size([1, 20480])
                audio = audio.detach().cpu().numpy().astype(np.float32)
                audio = audio_padding(audio,44100)
                output_dir = args.o + f'/{inst}/predicted/{num_idx}_{inst}_p.wav'
                print(output_dir)
                os.makedirs(os.path.dirname(output_dir), exist_ok=True) #(1, 18432)
                scipy.io.wavfile.write(output_dir, 44100, audio.T)
            for _, codes in enumerate([mixed_loops]):
                inst_codes = codes.permute(0,2,1) # torch.Size([1, 9, seq_len])
                latent = dac_model.quantizer.from_codes(inst_codes)[0]
                audio = dac_model.decode(latent)[0]
                audio = audio.detach().cpu().numpy().astype(np.float32)
                audio = audio_padding(audio,44100*4)
                output_dir = args.o + f'/{inst}/audio/{num_idx}_{inst}_a.wav'
                os.makedirs(os.path.dirname(output_dir), exist_ok=True) #(1, 18432)
                scipy.io.wavfile.write(output_dir, 44100, audio.T)
    return error_list




if __name__ == "__main__":
    data_dir = '/disk2/st_drums/check/'
    result_dir = '/disk2/st_drums/results/' #/drumonly/'
    audio_encoding_type = 'codes'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', type=str, default='kick', help='kick, snare, hihat')
    parser.add_argument('--i', type=str, required=True, help='input wav dir')
    parser.add_argument('--o', type=str, default='./results', help='output wav dir')
    # parser.add_argument('--wandb', type=bool, default=False, help='True, False')
    parser.add_argument('--layer_cut', type=int, default='1', help='enc(or dec)_num_layers // layer_cut')
    parser.add_argument('--dim_cut', type=int, default='1', help='enc(or dec)_num_heads, _d_model // dim_cut')
    parser.add_argument('--batch_size', type=int, default='1', help='batch size')
    args = parser.parse_args()
    
    ######### MAIN #############
    
    dac_only = True
    gpu = True

    BATCH_SIZE = 1
    NUM_WORKERS = 15
    ############################
    dac_model_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(dac_model_path) 
    dac_model.eval()
    if gpu :
        dac_model.cuda()
    
    if dac_only:
        config = InstDecoderConfig(audio_rep = audio_encoding_type, args = args)
        # model = InstDecoderModule(config, dac_model, 1)
        model = InstDecoderModule(config, 1)
    else:
        config = EncoderDecoderConfig(audio_rep = audio_encoding_type, args = args)
        model = EncoderDecoderModule(config)

    if args.train_type == 'kick':
        ckpt = torch.load('./checkpoints/kick.ckpt', map_location='cpu')
    elif args.train_type == 'snare':
        ckpt = torch.load('./checkpoints/snare.ckpt', map_location='cpu')
    elif args.train_type == 'hihat':
        ckpt = torch.load('./checkpoints/hihat.ckpt', map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    if gpu :
        model.cuda()
    model.eval() 

    ### real wav DATA 
    test_dataloader = dataset_wav_load(args.i, 1)

    ### Generate
    # export CUDA_VISIBLE_DEVICES=1
    error_list = inst_generate(test_dataloader, args.train_type, args)
    print('error occured : ', error_list)


# python eval_c.py 