from torch.utils.data import Dataset
import glob
import numpy
import pretty_midi
import os
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
import numpy as np
import pretty_midi
from einops import rearrange
import argparse
import matplotlib.pyplot as plt


class DrumSlayerDataset(Dataset):
    def __init__(self, split, args, audio_encoding_type = 'codes', max_len=152):
        assert audio_encoding_type in ["latents", "codes", "z"] # dim: 72, 9, 1024
        self.file_path = '/disk2/st_drums/check/'
        self.inst = args.train_type
        self.oneshot_path = '/disk2/st_drums/check/one_shot_npy/'
        self.split = split # train, valid, test
        self.max_len = max_len
        self.encoding_type = audio_encoding_type
        self.num_data = len(glob.glob(self.file_path + f"drum_data_{split}/mixed_loops/*.npy"))
        self.train_type = args.train_type
        
    def __getitem__(self, idx):
        if self.split == "valid":
            idx += 400000-30000
        elif self.split == "test":
            idx += 400000-30000-30000
                 
        audio_rep = np.load(self.file_path + f"drum_data_{self.split}/mixed_loops/{idx}_{self.encoding_type}.npy") ## dac npy 생성 -> preprocess_dac.py
        audio_rep_l = self.tokenize_inst(audio_rep,4) # MONO, (9, 345+8)
        audio_dac = rearrange(audio_rep_l, 'd t -> t d') # (345+8, 9)

        if self.inst == "kick" or self.inst == "snare": # 1 sec
            # one_shot_rep = np.load(self.oneshot_path+ f"{self.split}/{self.inst}_gen/{self.inst}_{idx}_{self.encoding_type}.npy") 
            one_shot_rep = np.load(self.oneshot_path+ f"{self.inst}/{self.inst}_{idx}_{self.encoding_type}.npy") 
            one_shot_rep = one_shot_rep[:,0:90]
            oneshot_rep_l = self.tokenize_inst(one_shot_rep,0) # MONO, (9, 173+8+2)
            oneshot_dac = rearrange(oneshot_rep_l, 'd t -> t d') # (173+8+2, 9)
            # x: (b,) 345+8, 9
            # y: (b,) 173+8+2, 9  = batch, 183
        elif self.inst == "hihat": # 0.5초
            one_shot_rep = np.load(self.oneshot_path+ f"{self.inst}/{self.inst}_{idx}_{self.encoding_type}.npy") ## TODO :hihat -> hhclosed
            one_shot_rep = one_shot_rep[:,0:40] # 9, seq
            oneshot_rep_l = self.tokenize_inst(one_shot_rep,0) # MONO, (9, 173+8+2)
            oneshot_dac = rearrange(oneshot_rep_l, 'd t -> t d') # (173+8+2, 9)
        elif self.inst == 'ksh':
            kick_shot_rep = np.load(self.oneshot_path+ f"kick/kick_{idx}_{self.encoding_type}.npy")
            snare_shot_rep = np.load(self.oneshot_path+ f"snare/snare_{idx}_{self.encoding_type}.npy")
            hihat_shot_rep = np.load(self.oneshot_path+ f"hihat/hihat_{idx}_{self.encoding_type}.npy")
            # hihat_shot_rep = np.load(self.oneshot_path+ f"{self.split}/hihat_gen/hihat_{idx}_{self.encoding_type}.npy")
            kick_shot_rep_l = self.tokenize_inst(kick_shot_rep,2) # MONO, (9, 173+8+2)
            snare_shot_rep_l = self.tokenize_inst(snare_shot_rep,2) # MONO, (9, 173+8+2)
            hihat_shot_rep_l = self.tokenize_inst(hihat_shot_rep,2) # MONO, (9, 173+8+2)
            kick_shot_dac = rearrange(kick_shot_rep_l, 'd t -> t d') # (173+8+2, 9)
            snare_shot_dac = rearrange(snare_shot_rep_l, 'd t -> t d') # (173+8+2, 9)
            hihat_shot_dac = rearrange(hihat_shot_rep_l, 'd t -> t d') # (173+8+2, 9)
            oneshot_dac = np.concatenate((kick_shot_dac[:-1], snare_shot_dac[:-1], hihat_shot_dac), axis=0) # (549-2,9)
            # x: (b,) 345+8, 9
            # y: (b,) 549-2, 9
        elif self.inst == 'kshm':
            pass

        return audio_dac, oneshot_dac#, idx

    def __len__(self):
        return self.num_data
    


    def tokenize_inst(self,inst_dac_l,length): # inst_dac_l : 9, seq (345 or 173)   
        
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

            all_tokens_np[i, start: end] = codes + 1 # +2000
        return all_tokens_np # (9,345)


if __name__ == "__main__":
    # data_dir = '/workspace/DrumSlayer/generated_data/'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train_type', type=str, default='hihat', help='ksh, kshm, kick, snare, hihat')
    # args = parser.parse_args()

    # dataset = DrumSlayerDataset("test", args)
    # x = dataset[0]
    # breakpoint()
    # pass
    data_dir = '/disk2/st_drums/check'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', type=str, default='kick', help='ksh, kshm, kick, snare, hihat')
    args = parser.parse_args()

    dataset = DrumSlayerDataset("test", args)


    