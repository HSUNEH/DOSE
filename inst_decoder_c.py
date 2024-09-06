import math
import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
import pl_bolts
from tqdm import tqdm
import numpy as np

from einops import rearrange


class InstDecoderConfig:
    def __init__(self, audio_rep, args, **kwargs):
        self.train_type = args.train_type
        
        # # Encoder hparams.
        # self.enc_num_layers = 10//args.layer_cut
        # self.enc_num_heads = 12//args.dim_cut
        # self.enc_d_model = 768//args.dim_cut
        # self.enc_d_ff = 3072
        # Decoder hparams.
        self.dec_num_layers = 10//args.layer_cut
        self.dec_num_heads = 16//args.dim_cut
        self.dec_d_model = 1024//args.dim_cut
        self.dec_d_ff = 3072
        # Dropout hparams.
        self.embed_dropout = 0.1
        self.attn_dropout = 0.1
        self.ff_dropout = 0.1
        # Other hparams.
        self.midi_vocab_size = 1000+128+4+3 ## <start>,<end>,<pad>,<sep> 1000 time bins, 128 velocity bins, 3 drum types. 1134

        self.audio_vocab_size = 1024+1   # 1024 dac tokens + <start>,<end>,<pad>
        # self.dac_vocab_size = 4+1000+128+1025 # 2157 # <start>,<end>,<pad>,<sep> 1000 time bins, 128 velocity bins, 3 drum types. + 1025 dac tokens
        # self.audio_rep = "latents" # "latents", "codes", "z"
        self.audio_rep = audio_rep
        if self.audio_rep == "latents":
            self.audio_rep_dim = 72
        elif self.audio_rep == "z":
            self.audio_rep_dim = 1024
        elif self.audio_rep == "codes":
            self.audio_rep_dim = 9
        else:
            raise ValueError("Invalid audio_rep type")
        # self.max_target_len = 345+355
        self.max_len = 345+8+345+8+2+10
        self.padding_loss_lambda = 0.1
        if self.train_type == 'kick' or self.train_type == 'snare' :
            self.inst_dac_length = 173
        elif self.train_type == 'hihat':
            self.inst_dac_length = 173 
        
        if self.train_type not in ['ksh', 'kick', 'snare', 'hihat', 'kshm'] :
            raise ValueError("Invalid train_type")
        

class InstDecoderModule(pl.LightningModule):
    def __init__(self, config, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.config = config
        self.inst_decoder = InstDecoder(config)
        # self.loss_function = self.loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim+1)] 
        if self.config.train_type == 'kshm':
            self.loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim+1)] 
        else: 
            self.total_loss = nn.CrossEntropyLoss()
            self.audio_loss = nn.CrossEntropyLoss()
            self.padding_loss = nn.CrossEntropyLoss()
            self.dac_loss= nn.CrossEntropyLoss()
            self.start_loss = nn.CrossEntropyLoss()
            # self.loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)] 
            # self.padding_loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)] 
            # self.padding_in_loss_functions_l = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)]
            # self.padding_in_loss_functions_r = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)]
    def training_step(self, batch, batch_idx):
        audio_losses, dac_losses, start_loss, mean_acc = self.step(batch,batch_idx)


        # self.log("train_total_loss", total_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # if self.config.train_type == 'kshm':
        self.log("train_audio_loss", audio_losses.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_dac_loss", dac_losses.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_start_loss", start_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_mean_acc", mean_acc.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return (start_loss + audio_losses)
    
    def validation_step(self, batch, batch_idx):
        audio_losses, dac_losses,start_loss, mean_acc = self.step(batch,batch_idx)
        # self.log("valid_total_loss", total_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # if self.config.train_type == 'kshm':
        self.log("valid_audio_loss", audio_losses.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("valid_dac_loss", dac_losses.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("valid_start_loss", start_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("valid_mean_acc", mean_acc.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return (start_loss + audio_losses)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return self.optimizer

    def step(self, batch, batch_idx):
        
        
        if self.config.train_type == 'kshm': #TODO : need to fixed (padding 뒤로 뺌) #kick snare hihat dac +  midi token
            pass
        
        # elif self.config.train_type == 'ksh': 
        #     x, y = batch
        
        #     # x : b, 345+8, 9
        #     # y : b, 547(183*3-2), 9

        #     pass

        else: #inst 
                    
            x, y = batch # x: audio dac, y: oneshot dac # TODO:여기 잘못 표기한건지 아님 잘못 된건지?
            
            # x: b, 345+8, 9
            # y: b, 173+8+2, 9
            
            x = x.cuda()
            y = y.cuda()

            pred = self.inst_decoder(x, y[:,:-1]) # (batch_size, seq_len, total_vocab_size) torch.Size([4, 1471, 10396])

            # y_pred : torch.Size([4, 2737, 2157]) b, t, v
            # y : torch.Size([4, 2738, 19])
            
            
            # x_l = x[:,:,0,:].long()
            x_l =x
            xy = torch.cat((x_l, y[:,:,:]), dim=1)
            
            # x_l :     b, 345, d
            # y :       b, 347, d 
            # xy :      b, 692, d
            # pred :    b, 691, d*v
            target = xy[:,1:,:].long() # b t d 
            
            #### Audio Dac #### condition
            dac_losses = []
            dac_logits = pred[:,:352,:] # b s d*v
            dac_logits = dac_logits.view(dac_logits.shape[0], dac_logits.shape[1], 9, 1025)
            dac_logits = rearrange(dac_logits, 'b s d v -> b v s d')
            dac_target = target[:,:352,:] # b s d 
            dac_losses.append(self.dac_loss(dac_logits, dac_target.long()))

            #### Contents Start ####
            start_losses = []
            start_logits = pred[:,352:352+8+2,:] # b, s, d*v
            start_logits = start_logits.view(start_logits.shape[0], start_logits.shape[1],9, 1025) # b, s, d, v
            start_logits = rearrange(start_logits, 'b s d v -> b v s d') # b, v, s, d
            start_target = target[:,352:352+8+2,:] # b, s, d
            start_losses.append(self.start_loss(start_logits, start_target.long()))
            
            #### Contents #### 
            audio_losses = []
            audio_logits = pred[:,352:352+self.config.inst_dac_length+8+2,:]
            audio_logits = audio_logits.view(audio_logits.shape[0],audio_logits.shape[1],9, 1025)
            audio_logits = rearrange(audio_logits, 'b s d v -> b v s d')
            audio_target = target[:,352:352+self.config.inst_dac_length+8+2,:] # b s d
            audio_losses.append(self.audio_loss(audio_logits, audio_target.long()))
            
            
            # Calculate Accuracy.

            audio_accs = []
            acc_logits = pred[:,352:352+self.config.inst_dac_length+8+2,:]
            acc_logits = acc_logits.view(acc_logits.shape[0],acc_logits.shape[1],9, 1025)
            acc_logits = rearrange(acc_logits, 'b s d v -> b v s d')
            acc_target = target[:,352:352+self.config.inst_dac_length+8+2,:] # s d
            acc_pred = torch.argmax(acc_logits, dim=1)
            audio_acc = torch.sum(acc_pred == acc_target).float() / torch.numel(acc_target)
            audio_accs.append(audio_acc)

        dac_loss = sum(dac_losses)/len(dac_losses)
        start_loss = sum(start_losses)/len(start_losses)
        audio_loss = sum(audio_losses)/len(audio_losses)
        mean_acc = (sum(audio_accs)) /len(audio_accs)
        
        return audio_loss, dac_loss, start_loss, mean_acc



    def forward(self, batch): # evaluation step
        
        x, y= batch
        # y = rearrange(y, 'b v t -> b t v') # (4, 1472, 10)
        
        # x : torch.Size([1, 345, 2, 9])
        # y : torch.Size([1, 9, 355])
        # y : torch.Size([1, 355, 9])
         
        # ## Training Check ##
        # y = rearrange(y, 'b v t -> b t v') # (4, 1472, 10)
        # y_pred = self.inst_decoder(x, y[:,:-1]) # (batch_size, seq_len, total_vocab_size) torch.Size([4, 1471, 10396])
        

        ###### For Inference ######
        y_pred,end = self.inst_decoder(x)
        # y_pred : torch.Size([1, 355, 9])


        return (y_pred,end)

class InstDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.positional_encoder = PositionalEncoder(self.config.dec_d_model, self.config.embed_dropout) # 768, 0.1

        # self.decoder = Decoder(config)
        # self.softmax = nn.Softmax(dim=-1)
        # self.embedding = nn.Embedding(1025, 1)
        if self.config.train_type == 'kshm':
            self.dac_projection_layer = nn.Linear(config.dec_d_model, config.midi_vocab_size + config.audio_vocab_size * 9) ##number
        else: 
            self.dac_projection_layer = nn.Linear(config.dec_d_model,  config.audio_vocab_size * 9)
        self.midi_embedding_layer = nn.Embedding(config.midi_vocab_size, config.dec_d_model)
        self.audio_embedding_layer = nn.ModuleList([nn.Embedding(1025, config.dec_d_model) for _ in range(9)]) 
        self.dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.dec_num_layers)]
        )
        self.ln = nn.LayerNorm(config.dec_d_model)
        self.fc = nn.Linear(config.dec_d_model, 9*config.audio_vocab_size)
    def forward(self, x_l, y=None, strategy="greedy", sample_arg=None): # strategy : top-p, greedy

        ### inference ###
        if y is None:
            if self.config.train_type == 'kick' or self.config.train_type == 'snare' or self.config.train_type == 'hihat':
                seq_len =  1+self.config.inst_dac_length+8+1
            elif self.config.train_type == 'ksh':
                seq_len = 183*3-2
            elif self.config.train_type == 'kshm':
                pass
            return self.generate(x_l, self.config.train_type, seq_len =  seq_len, strategy=strategy, sample_arg=sample_arg)
        # x_l : torch.Size([16, 353, 9])
        # y   : torch.Size([16, 182, 9])
        
        
        ### Train ###
        
        if self.config.train_type == 'kshm':
            pass
        else: 
        # Input (ksh)
        # x_l :  b, 353, 9
        # y :    b, 546, 9
        # Input (inst)
        # x_l :  b, 353, 9
        # y :    b, 182, 9
            ### train ###
            for i in range(9):
                x_tok_embedding = self.audio_embedding_layer[i](x_l[:,:,i])  # 
                y_tok_embedding = self.audio_embedding_layer[i](y[:,:,i])  # y[:,:,i] = [batch_size, seq_len, 1] -> torch.Size([batch_size, seq_len, dec_d_model])
                if i == 0:
                    audio_tok_embedding = x_tok_embedding
                    inst_tok_embedding = y_tok_embedding
                audio_tok_embedding = audio_tok_embedding + x_tok_embedding
                inst_tok_embedding = inst_tok_embedding + y_tok_embedding

            embedding = torch.cat((audio_tok_embedding, inst_tok_embedding), dim =1) 

            
            # tok_embedding.shape == (batch_size, seq_len, embed_dim)
            output = self.dropout(embedding)
            output = self.blocks(output)
            output = self.ln(output)
            output = self.fc(output) #torch.Size([16, 535, 9225]) / ksh : torch.Size([16, 899, 9225])
            # output.shape == (batch_size, seq_len, vocab_size)
            return output


    def generate(self, x_l, train_type, seq_len, strategy, sample_arg=None):
        # Input
        # enc_output: [batch_size, seq_len, d_model]
        # Output
        # midi_logits: [batch_size, seq_len, midi_vocab_size]
        
        batch_size = x_l.shape[0]

        # Start token.
        end = False
        count = 0
        for i in tqdm(range(seq_len)):
            for j in range(9): 
                x_tok_embedding = self.audio_embedding_layer[j](x_l[:,:,j])  # 
                if j == 0:
                    audio_tok_embedding = x_tok_embedding
                audio_tok_embedding = audio_tok_embedding + x_tok_embedding
            embedding = audio_tok_embedding #torch.cat((audio_tok_embedding, torch.zeros(batch_size, 1, 9).long()), dim =1) 

            output = self.dropout(embedding)
            output = self.blocks(output)
            output = self.ln(output)
            output = self.fc(output) #1,9,9225 torch.Size([16, 699, 9252])  

            # sampling
            sampled_token = self.sample(output[:,352+i,:], strategy=strategy, sample_arg=sample_arg) #[batch_size, 1, 9 or 10]
            if train_type == 'kshm':
                pass
            elif train_type == 'ksh':
                if torch.all(sampled_token == 0) and i != 0:
                    x_l = torch.cat([x_l, sampled_token], dim=1)
                    end = True
                    count+=1
                    if count == 2:
                        return x_l, end
                else:
                    x_l = torch.cat([x_l, sampled_token], dim=1)
            else:
                if torch.all(sampled_token == 0) and i != 0:
                    x_l = torch.cat([x_l, sampled_token], dim=1)
                    end = True
                    return x_l, end
                else:
                    x_l = torch.cat([x_l, sampled_token], dim=1)

        return x_l, end
    
    def sample(self, logits, strategy, sample_arg=None):
        # Input
        # logits: [batch_size, midi_vocab_size]
        # Output
        # sampled_token: [batch_size, 1]
        if strategy == "top-p":
            p = 0.9
            sampled_indices =[]
            if self.config.train_type == 'kshm':
                midi_logits = logits[:, :self.config.midi_vocab_size]
                midi_sorted_logits, midi_indices = torch.sort(midi_logits, descending=True)
                midi_prob = torch.softmax(midi_sorted_logits, dim=-1)
                midi_cum_prob = torch.cumsum(midi_prob, dim=-1)
                midi_index = torch.where(midi_cum_prob >= p, torch.ones_like(midi_cum_prob), torch.zeros_like(midi_cum_prob)).argmax(dim=-1)
                midi_sorted_logits[0,midi_index+1:] = -float("inf")
                midi_top_p_prob = torch.softmax(midi_sorted_logits, dim=-1)
                midi_sampled = torch.multinomial(midi_top_p_prob, 1)
                midi_reindex = torch.gather(midi_indices, dim=-1, index=midi_sampled)
                sampled_indices.append(midi_reindex)

            for i in range(9):
                if self.config.train_type == 'kshm':
                    audio_logits = logits[:, self.config.midi_vocab_size + i*self.config.audio_vocab_size : self.config.midi_vocab_size + (i+1)*self.config.audio_vocab_size]
                else:
                    audio_logits = logits[:, i*self.config.audio_vocab_size : (i+1)*self.config.audio_vocab_size]
                audio_sorted_logits, audio_indices = torch.sort(audio_logits, descending=True)
                audio_prob = torch.softmax(audio_sorted_logits, dim=-1)
                audio_cum_prob = torch.cumsum(audio_prob, dim=-1)
                audio_index = torch.where(audio_cum_prob > p, torch.ones_like(audio_cum_prob), torch.zeros_like(audio_cum_prob)).argmax(dim=-1)
                audio_sorted_logits[0,audio_index+1:] = -float("inf")
                audio_top_p_prob = torch.softmax(audio_sorted_logits, dim=-1)
                audio_sampled = torch.multinomial(audio_top_p_prob, 1)
                audio_reindex = torch.gather(audio_indices, dim=-1, index=audio_sampled)
                sampled_indices.append(audio_reindex)

            sampled_indices = torch.cat(sampled_indices, dim=-1).unsqueeze(0)

            return sampled_indices #torch.Size([1, 1, 9])
        if strategy == "greedy":
            # logits : torch.Size([1, 10396])
            sampled_indices =[]
     
            for i in range(9):
                audio_logits = logits[:, i*self.config.audio_vocab_size : (i+1)*self.config.audio_vocab_size] #torch.Size([1, 1029])
                audio_reindex = torch.argmax(audio_logits, dim=-1).unsqueeze(-1)
                sampled_indices.append(audio_reindex)

            sampled_indices = torch.cat(sampled_indices, dim=-1).unsqueeze(0)
            return sampled_indices #torch.Size([1, 1, 9])
        
        # elif strategy == "top-k": # This is generated by copilot. Need to check.
        #     k = sample_arg
        #     top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        #     top_k_probs = self.softmax(top_k_logits)
        #     sampled_token = torch.multinomial(top_k_probs, 1)
        #     return top_k_indices.gather(-1, sampled_token)
        # elif strategy == "top-p": # This is generated by copilot. Need to check.
        #     p = sample_arg
            
        #     sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        #     cumulative_probs = torch.cumsum(self.softmax(sorted_logits), dim=-1)
        #     sorted_indices_to_remove = cumulative_probs > p
        #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        #     sorted_indices_to_remove[..., 0] = 0
        #     indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        #     logits = logits.masked_fill(indices_to_remove, -float('inf'))
        #     sampled_token = torch.multinomial(self.softmax(logits), 1)
        #     return sampled_token

  
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.dec_d_model
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(config.ff_dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.dec_d_model
        self.num_heads = config.dec_num_heads
        assert embed_dim % self.num_heads == 0, "invalid heads and embedding dimension configuration"
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj_dropout = nn.Dropout(config.ff_dropout)
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.max_len, config.max_len).bool())
            .unsqueeze(0).unsqueeze(0)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # x.shape == (batch_size, seq_len, embed_dim)
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1,2)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape == (batch_size, num_heads, seq_len, head_dim)

        mask = self.mask[:, :, :seq_len, :seq_len]
        y = F.scaled_dot_product_attention(q, k_t, v, is_causal=True)
        
        y = y.transpose(1, 2)
        # y.shape == (batch_size, seq_len, num_heads, head_dim)
        y = y.reshape(batch_size, seq_len, -1)
        # y.shape == (batch_size, seq_len, embed_dim)
        y = self.proj_dropout(self.proj(y))
        return y

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer("pe", self.get_positional_encoding())
    
    def get_positional_encoding(self):
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
        re
    def forward(self, x):
        # Input
        # x: [batch_size, seq_len, d_model]
        # Output
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


# class Encoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.enc_num_layers)])
    
#     def forward(self, x):
#         # Input
#         # x: [batch_size, seq_len, d_model]
#         # Output
#         # x: [batch_size, seq_len, d_model]
#         for layer in self.layers:
#             x = layer(x)
   
#         return x

# class EncoderLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.config = config
#         embed_dim = config.enc_d_model
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=config.enc_num_heads, dropout=config.attn_dropout,batch_first=True, norm_first=True)
#         # ff_dim = config.enc_d_ff
#         # self.ln1 = nn.LayerNorm(embed_dim)
#         # self.ln2 = nn.LayerNorm(embed_dim)
#         # self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=config.enc_num_heads, dropout=config.attn_dropout, batch_first=True)
#         # self.ff = nn.Sequential(
#         #     nn.Linear(embed_dim, ff_dim),
#         #     nn.GELU(),
#         #     nn.Linear(ff_dim, embed_dim),
#         #     nn.Dropout(config.ff_dropout),
#         # )
    
#     def forward(self, x):
#         # # training2 # 원본
#         # x = self.ln1(x)
#         # x = x + self.mha(x,x,x)[0]
#         # x = x + self.ff(self.ln2(x))
    
#         # # training
#         # residual = x
#         # x = residual + self.mha(x,x,x)[0]
#         # x = self.ln1(x)
        
#         # residual = x
#         # x = residual + self.ff(x)
#         # x= self.ln2(x)
        
#         # # 정석으로 바꿔봤다
#         x = self.encoder_layer(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.dec_num_layers)])
    
#     def forward(self, x, enc_output):
#         # Input
#         # x: [batch_size, seq_len, d_model]
#         # enc_output: [batch_size, seq_len, d_model]
#         # Output
#         # x: [batch_size, seq_len, d_model]
#         for layer in self.layers:
#             x = layer(x, enc_output)
#         return x

# class DecoderLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         embed_dim = config.dec_d_model
#         self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=config.dec_num_heads, dropout=config.attn_dropout, batch_first=True, norm_first=True)
#         # ff_dim = config.dec_d_ff
        
#         # self.ln1 = nn.LayerNorm(embed_dim)
#         # self.ln2 = nn.LayerNorm(embed_dim)
#         # self.ln3 = nn.LayerNorm(embed_dim)
#         # self.mha1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=config.dec_num_heads, dropout=config.attn_dropout, batch_first=True)
#         # self.mha2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=config.dec_num_heads, dropout=config.attn_dropout, batch_first=True)
#         self.register_buffer("causal_mask", torch.triu(torch.ones(config.max_target_len, config.max_target_len).bool(), diagonal=1))
#         # self.ff = nn.Sequential(
#         #     nn.Linear(embed_dim, ff_dim),
#         #     nn.GELU(),
#         #     nn.Linear(ff_dim, embed_dim),
#         #     nn.Dropout(config.ff_dropout),
#         # )
    
#     def forward(self, x, enc_output):
#         x = self.decoder_layer(x,enc_output,tgt_mask=self.causal_mask[:x.size(1),:x.size(1)] ,tgt_is_causal=True)
#         # x = self.ln1(x)
#         # x = x + self.mha1(x, x, x, is_causal=True, attn_mask=self.causal_mask[:x.size(1),:x.size(1)])[0]
#         # x = x + self.mha2(self.ln2(x), enc_output, enc_output, is_causal=False)[0]
#         # x = x + self.ff(self.ln3(x))
#         return x    


if __name__ == "__main__":
    print('good')
