import torch

from packaging import version
import importlib.metadata
from transformers import Phi3Config, Phi3ForCausalLM, Phi3Model, Phi3PreTrainedModel
from transformers.models.phi3.modeling_phi3 import (
    Phi3DecoderLayer,
    Phi3Attention,
    Phi3FlashAttention2,
    Phi3SdpaAttention,
    Phi3MLP,
    Phi3RMSNorm,
    Phi3RotaryEmbedding
)

from torch import nn
from transformers.utils import logging

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils.import_utils import _is_package_available

from peft import PeftModel

logger = logging.get_logger(__name__)


def is_transformers_attn_greater_or_equal_4_38():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.38.0"
    )


def is_transformers_attn_greater_or_equal_4_40():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.40.0"
    )
class ModifiedPhi3Attention(Phi3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False
class ModifiedPhi3FlashAttention2(Phi3FlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

class ModifiedPhi3SdpaAttention(Phi3SdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

PHI3_ATTENTION_CLASSES = {
    "eager": ModifiedPhi3Attention,
    "flash_attention_2": ModifiedPhi3FlashAttention2,
    "sdpa": ModifiedPhi3SdpaAttention,
}

class ModifiedPhi3DecoderLayer(Phi3DecoderLayer):
    def __init__(self, config: Phi3Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = PHI3_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = Phi3MLP(config)
        self.input_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = Phi3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Phi3BiModel(Phi3Model):
    _no_split_modules = ["ModifiedPhi3DecoderLayer"]

    def __init__(self, config: Phi3Config):
        if not is_transformers_attn_greater_or_equal_4_38():
            raise ValueError(
                "The current implementation of Phi3EncoderModel follows modeling_phi3.py of transformers version >= 4.38.0"
            )
        Phi3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [
                ModifiedPhi3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.rotary_emb = Phi3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_seen_tokens=None,
        output_attentions=False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        raise NotImplementedError("Phi3BiModel does not support causal masking")


class Phi3BiForMNTP(Phi3ForCausalLM):
    def __init__(self, config):
        Phi3PreTrainedModel.__init__(self, config)
        self.model = Phi3BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # getter for PEFT model
    def get_model_for_peft(self):
        return self.model

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    # save the PEFT model
    def save_peft_model(self, path):
        self.model.save_pretrained(path)





