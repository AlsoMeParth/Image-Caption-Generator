import os
if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
from src.model import ImageCaptioningModel
from transformers import AutoTokenizer, AutoImageProcessor
import torch
from PIL import Image
from pathlib import Path
import argparse


def load_processor():
    image_processor = AutoImageProcessor.from_pretrained( 
        "google/vit-base-patch16-224-in21k"
        )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return image_processor, tokenizer


def load_model(checkpoint, device):
    model = ImageCaptioningModel()

    checkpoint = torch.load(
        checkpoint,
        map_location = "cpu",
        weights_only = True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model = model.to(device)

    model.eval()

    print(
        f"Loaded checkpoint from Epoch: {checkpoint['epoch']}"
        f"With Validation Loss: {checkpoint['val_loss']:.4f}"
        )
    
    return model

@torch.inference_mode()
def generate_caption(
    model,
    image,
    image_processor,
    tokenizer,
    device
):
    pixel_values = image_processor(
        images = image.convert("RGB"),
        return_tensors = "pt",
    )["pixel_values"].to(device)

    encoder_outputs = model.encoder(
        pixel_values = pixel_values
    )

    image_features = encoder_outputs.last_hidden_state
    image_features = model.visual_projection(image_features)

    start_token = torch.full(
        (1, 1),
        tokenizer.eos_token_id,
        dtype=torch.long,
        device=device
    )

    generated_ids = model.decoder.generate(
        input_ids=start_token,
        encoder_hidden_states=image_features,
        max_new_tokens=24,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    caption=tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    ).strip()

    return caption

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("best_model.pt")
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    image_processor, tokenizer = load_processor()

    model = load_model(
        args.checkpoint,
        device
    )
    with Image.open(args.image) as image:
        caption = generate_caption(
            model,
            image,
            image_processor,
            tokenizer,
            device
        )

    print(f"Caption: {caption}")

if __name__ == "__main__":
    main()