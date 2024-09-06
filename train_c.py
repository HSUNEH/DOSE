# from encoder_decoder_inst_c import EncoderDecoderModule, EncoderDecoderConfig    
from inst_decoder_c import InstDecoderModule, InstDecoderConfig
from dataset import DrumSlayerDataset
import numpy as np
import dac
from torch.utils.data import DataLoader
import wandb
import lightning.pytorch as pl
import os
import datetime
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import argparse
from midi_2_wav.midi_2_wav import SingleShot, MIDI, Loop, OtherSound,OtherLoop,VocalLoop, AllData

audio_encoding_type = "codes" # "latents", "codes", "z" (dim: 72, 9, 1024)

# data_dir = '/workspace/DrumSlayer/generated_data/'
data_dir = '/disk2/st_drums/check/'
# trained_dir = '/workspace/ckpts'
trained_dir = '/disk2/st_drums/ckpts'

def main(args):
    # model_path = dac.utils.download(model_type="44khz")
    # dac_model = dac.DAC.load(model_path)  
    # dac_model = dac_model.cuda()
    BATCH_SIZE = args.batch_size 
    EXP_NAME = f"{datetime.datetime.now().strftime('%m-%d-%H-%M')}-STDT-{args.train_type}-{args.layer_cut}_{args.dim_cut}_{args.batch_size}"
    os.makedirs(f"{trained_dir}/{EXP_NAME}/", exist_ok=True)

    if args.wandb == True: 
        wandb.init(project="DrumTranscriber", name=EXP_NAME)
        WANDB = True
    else:
        WANDB = False    
    train_dataset = DrumSlayerDataset("train", args)
    valid_dataset = DrumSlayerDataset("valid", args)
    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_dataloader = DataLoader(valid_dataset,batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    

    if decoder_only:
        config = InstDecoderConfig(audio_rep = audio_encoding_type, args = args)
        model = InstDecoderModule(config,batch_size = BATCH_SIZE)
    # else:
    #     config = EncoderDecoderConfig(audio_rep = audio_encoding_type, args = args)
    #     model = EncoderDecoderModule(config,dac_model)
    

    #### LOAD PRETRAINED MODEL
    # ckpt_dir = '/workspace/ckpts/08-15-12-27-STDT-hihat-1_1_16/train_audio_loss=0.95-valid_audio_loss=3.41-step=213334.ckpt'    
    # ckpt_dir = '/workspace/ckpts/05-24-01-18-STDT-snare-1_1_16/train_audio_loss=0.46-valid_audio_loss=0.99-step=358334.ckpt'
    # ckpt = torch.load(ckpt_dir, map_location='cpu')
    # model.load_state_dict(ckpt['state_dict'])
    
    ddp_strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)

    
    check_point_n_steps = 5000 # 8 gpu -> 5000
    valid_n_steps = check_point_n_steps
    n_step_checkpoint = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor="train_audio_loss",
        mode="min",
        dirpath=f"{trained_dir}/{EXP_NAME}/",
        filename = "{train_audio_loss:.2f}-{valid_audio_loss:.2f}-{step}", 
        every_n_epochs = 1
        # every_n_train_steps=check_point_n_steps, # n_steps
    )
    
    n_step_earlystop = pl.callbacks.EarlyStopping(                                                                                                                                                                    
                        monitor="valid_audio_loss",                                                                                                                                                                        
                        min_delta=0.00,                                                                                                                                                                            
                        patience=15,                                                                                                                                                                                
                        verbose=True,                                                                                                                                                                              
                        mode="min",                                                                                                                                                                                
                        check_on_train_epoch_end=False,                                                                                                                                                            
                    )                                                                                                                                                                                          

    if WANDB:
        logger = WandbLogger(name=EXP_NAME, project="DrumSlayer")
        # trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=5, precision='16-mixed', callbacks=[n_step_checkpoint, n_step_earlystop], strategy=ddp_strategy, val_check_interval=valid_n_steps)
        # trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=150, precision='16-mixed', callbacks=[n_step_checkpoint], strategy=ddp_strategy, val_check_interval=valid_n_steps)
        trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=150, precision='16-mixed', callbacks=[n_step_checkpoint, n_step_earlystop], strategy=ddp_strategy,check_val_every_n_epoch=1  ) #val_check_interval = valid_n_steps)
    else:
        # logger = TensorBoardLogger(save_dir=f"{trained_dir}/{EXP_NAME}/logs", name=EXP_NAME)
        # trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=5, precision='16-mixed', callbacks=[n_step_checkpoint, n_step_earlystop], strategy=ddp_strategy, val_check_interval=valid_n_steps)
        trainer = pl.Trainer(accelerator="gpu", devices=NUM_DEVICES, max_epochs=150, precision='16-mixed',  callbacks=[n_step_checkpoint, n_step_earlystop], strategy=ddp_strategy, check_val_every_n_epoch=1  )


    trainer.fit(model=model, train_dataloaders=train_dataloader,val_dataloaders =valid_dataloader)
        

if __name__ == "__main__":
    
    decoder_only = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', type=str, default='kick', help='ksh, kshm, kick, snare, hihat')
    parser.add_argument('--wandb', type=bool, default=True, help='True, False')
    parser.add_argument('--layer_cut', type=int, default='1', help='enc(or dec)_num_layers // layer_cut')
    parser.add_argument('--dim_cut', type=int, default='1', help='enc(or dec)_num_heads, _d_model // dim_cut')
    parser.add_argument('--batch_size', type=int, default='16', help='batch size')
    args = parser.parse_args()
    
    NUM_DEVICES = [1,2,3,4,5,7]
    NUM_WORKERS = 15
    main(args)
    #export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
