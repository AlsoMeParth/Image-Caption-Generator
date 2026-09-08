import torch.nn as nn
from transformers import ViTModel, GPT2Config, GPT2LMHeadModel

class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        encoder_name = "google/vit-base-patch16-224-in21k",
        decoder_name = "gpt2"
    ):
        super().__init__()
        self.encoder = ViTModel.from_pretrained(encoder_name)

        decoder_config = GPT2Config.from_pretrained(decoder_name)
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True

        self.decoder = GPT2LMHeadModel.from_pretrained(
            decoder_name,
            config = decoder_config
        )

        encoder_dim = self.encoder.config.hidden_size
        decoder_dim = self.decoder.config.n_embd
        self.visual_projection = (
            nn.Identity()
            if encoder_dim == decoder_dim
            else nn.Linear(encoder_dim, decoder_dim)
        )

        self.decoder.config.pad_token_id = self.decoder.config.eos_token_id
    
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        encoder_outputs = self.encoder(pixel_values = pixel_values)
        
        image_features = encoder_outputs.last_hidden_state
        image_features = self.visual_projection(image_features)

        outputs = self.decoder(
            input_ids = input_ids,
            attention_mask = attention_mask,
            encoder_hidden_states = image_features,
            labels = labels
        )
        return outputs
    
    def freeze_for_alignment(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        
        for name, parameter in self.decoder.named_parameters():
            if "crossattention" in name or "ln_cross_attn" in name:
                parameter.requires_grad = True
        
        for parameter in self.visual_projection.parameters():
            parameter.requires_grad = True