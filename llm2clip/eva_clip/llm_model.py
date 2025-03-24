import torch
from torch import nn
from llm2vec import LLM2Vec

class LLM2VecTextTransformer(nn.Module):
    def __init__(self, text_proj=None):
        super().__init__()
        enable_bidirectional = True
        base_model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
        extra_model_name_or_path = None
        peft_path = "checkpoints/LLM2CLIP-Llama-3.1-8B"
        self.text = LLM2Vec.from_pretrained(
                base_model_name_or_path,
                peft_path,
                merge_peft = True,
                extra_model_name_or_path=extra_model_name_or_path,
                enable_bidirectional=enable_bidirectional,
                attn_implementation = "flash_attention_2",
                torch_dtype=torch.bfloat16
            )
        self.text_proj = text_proj
        
    def lock(self, **kwargs):
        for param in self.text.parameters():
            param.requires_grad = False
            
    def forward(self, text, batch_size=32): 
        with torch.autocast("cuda"):        
            x = self.text.encode(text,batch_size=batch_size).to(torch.float16)
            if self.text_proj is not None:
                x = self.text_proj(x, l2_norm=False)
        return x
    
    def set_grad_checkpointing(self, enable=True):
        #Not implemented
        pass 