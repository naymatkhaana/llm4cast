import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, PositionalEmbedding
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from transformers import AutoConfig
from peft import LoraConfig, PeftModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "llama-2-7b-miniguanaco"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}


# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)



# # Load LLaMA tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """
    def __init__(self, configs): #, device):
        super().__init__()
        self.task_name = configs.task_name
        self.is_gpt = 1 #configs.is_gpt
        self.patch_len = configs.patch_len
        self.pretrain = 1 #configs.pretrain
        self.stride = configs.stride
        self.pretrained = configs.use_pretrained

        print("################################################################### self.pretrained:", self.pretrained)
        print("################################################################### patch_len:", configs.patch_len)
        print("################################################################### stride:", configs.stride)

        self.patch_num = (configs.seq_len - self.patch_len) // self.stride + 1
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = self.stride
        patch_len = self.patch_len
        configs.d_model = 4096 #768
        self.d_model =  configs.d_model

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, self.stride, padding, configs.dropout)

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 

        self.mask_padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 

        self.patch_num += 1
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

                # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / self.stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

        ####################################################################################################################
        ####################################################################################################################
        


        if 1==1: # configs.is_gpt:
            if 1==1: #configs.pretrain:
                # self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
                # Load base model

                print("**************************************************************************************************** AutoModelForCausalLM.from_pretrained")
                self.gpt2 = AutoModelForCausalLM.from_pretrained(
                    "TencentARC/LLaMA-Pro-8B", #"meta-llama/Meta-Llama-3-8B", #"TencentARC/LLaMA-Pro-8B",
                    low_cpu_mem_usage=True, #model_name,
                    cache_dir="/scratch/fs47816/workdir/sample_scripts/time_series_dl/Time-Series-Library/models_cache",
                    quantization_config=bnb_config,
                    token="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",
                    device_map=device_map
                ) #if self.pretrained == 1 else AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("TencentARC/LLaMA-Pro-8B",low_cpu_mem_usage=True,quantization_config=bnb_config,token="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",device_map=device_map ))
                self.gpt2.config.use_cache = False
                self.gpt2.config.pretraining_tp = 1
                self.gpt2.config.output_hidden_states = True

            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.model.layers = self.gpt2.model.layers[: configs.gpt_layers] #[:6]
            print("gpt2 = {}".format(self.gpt2))

        #self.position_embedding = PositionalEmbedding(self.d_model)
        self.feature_projection = nn.Linear(self.d_model, self.d_model)
        self.binary_indicator_embedding = nn.Linear(self.patch_len,self.d_model)
        self.gate_w1 = nn.Linear(self.d_model, self.d_model)
        self.gate_w2 = nn.Linear(self.d_model, self.d_model)
        self.gate_sigmoid = nn.Sigmoid()
        self.ts_embed_dropout = nn.Dropout(configs.dropout)

        self.in_layer = nn.Linear(self.patch_len, configs.d_model) #configs.patch_len, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len) #1024 ) 
        #self.out_layer2 = nn.Linear(1024,configs.pred_len)
        
        # if 1==1 and 1==1: #configs.freeze and configs.pretrain:
        #     for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #         if 'ln' in name or 'wpe' in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False

        for layer in (self.in_layer, self.out_layer): #(self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, seq_x_mask):

        



        B, L, M = x_enc.shape
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        
     


        # print("************************************* Line 137: enc_out.shape",x_enc.shape )
        # # # do patching and embedding
        # x_enc = x_enc.permute(0, 2, 1)
        # # u: [bs * nvars x patch_num x d_model]
        # enc_out, n_vars = self.patch_embedding(x_enc)
        ######################################################## ENCODER
        #enc_out, attns = self.encoder(enc_out)

        x_enc = rearrange(x_enc, 'b l m -> b m l')
        x_enc = self.padding_patch_layer(x_enc)
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        enc_out = rearrange(x_enc, 'b m n p -> (b m) n p')


        # print("************************************* Line 138: enc_out.shape",enc_out.shape )

        #pos_enc = self.position_embedding(enc_out)
        enc_out = self.in_layer(enc_out) # + pos_enc
 
        #''''
        # mask processing
        #seq_x_mask= rearrange(seq_x_mask, 'b l m -> b m l')
        #seq_x_mask = self.mask_padding_patch_layer(seq_x_mask)
        #seq_x_mask = seq_x_mask.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        #seq_x_mask = rearrange(seq_x_mask, 'b m n p -> (b m) n p')
        #mask_embed = self.binary_indicator_embedding(seq_x_mask)

        # gated fusion with mask
        #gate = self.gate_sigmoid(self.gate_w1(enc_out) + self.gate_w2(mask_embed))
        #enc_out = gate * enc_out + (1 - gate) * mask_embed
        #enc_out = self.feature_projection(enc_out)

        

        #positional embedding encoding

        enc_out = self.ts_embed_dropout(enc_out)

        enc_out, attns = self.encoder(enc_out)

        if self.is_gpt:
            #print("Yesss here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", enc_out.shape)
            enc_out = self.gpt2(inputs_embeds=enc_out).hidden_states[-1] #.last_hidden_state

            # print("************************************* Line 145: dec_out.shape",enc_out.shape )
            
            # z: [bs x nvars x patch_num x d_model]
        #     enc_out = torch.reshape( enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        #     # # z: [bs x nvars x d_model x patch_num]
        #     enc_out = enc_out.permute(0, 1, 3, 2)
        # # Decoder
        # dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        # dec_out = dec_out.permute(0, 2, 1)

        dec_out =  self.out_layer(enc_out.reshape(B*M, -1)) 
        #dec_out = F.relu(self.out_layer(enc_out.reshape(B*M, -1)) )
        #dec_out = self.out_layer2(dec_out)
        dec_out = rearrange(dec_out, '(b m) l -> b l m', b=B)


        # print("************************************* Line 151: dec_out.shape",dec_out.shape )

        # dec_out = dec_out.permute(0, 2, 1)


        # outputs = outputs * stdev
        # outputs = outputs + means

                # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out *         #           (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = dec_out +         #           (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        dec_out = dec_out * stdev
        dec_out = dec_out + means


        # print("************************************* Line 166: dec_out.shape",dec_out.shape )

        return dec_out


        # return outputs

        """
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
        """     

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        seq_x_mask = mask
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, seq_x_mask)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
