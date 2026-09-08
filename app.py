import torch
import gradio as gr
from src.dataset import load_processor
from src.predict import load_model, generate_caption

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "src/best_model.pt"
image_processor, tokenizer = load_processor()
model = load_model(CHECKPOINT_PATH, device)

def caption_image(image):
    if image is None:
        gr.Error("Please upload an image.")
    else:
        return generate_caption(
            model,
            image,
            image_processor,
            tokenizer,
            device
        )

demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(
        type='pil',
        label="Upload an Image"
    ),
    outputs=gr.Textbox(
        label="Generated Caption"
    ),
    title="Image Caption Generator",
    description="Upload an image and let the model generate a caption."
)

if __name__ == "__main__":
    demo.launch()