import os
if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")

from dataset import load_pairs, split_by_image, load_processor
from collections import defaultdict
from nltk.tokenize import wordpunct_tokenize
from nltk.translate.bleu_score import corpus_bleu
from predict import generate_caption, load_model
import torch
from tqdm.auto import tqdm


CHECKPOINT = "best_model.pt"
CAPTIONS = "../Flickr8k/captions.txt"
IMAGES = "../Flickr8k/Images"

references_by_image = defaultdict(list)

pairs = load_pairs(CAPTIONS, IMAGES)
_, _, test = split_by_image(pairs, 0.1, 0.1, seed=42)

for image_path, caption in  test:
    references_by_image[image_path].append(caption)

def tokenize_caption(text):
    return wordpunct_tokenize(text.lower())

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = load_model(CHECKPOINT, device)
image_processor, tokenizer = load_processor()


all_references = []
all_predictions = []

for image_path, captions in tqdm(
    references_by_image.items(),
    desc = "Generating test captions"
):
    prediction = generate_caption(
        model, 
        image_path,
        image_processor,
        tokenizer,
        device
    )

    all_references.append(
        [
            tokenize_caption(caption)
            for caption in captions
        ]
    )

    all_predictions.append(
            tokenize_caption(prediction)
    )

bleu_1 = corpus_bleu(
    all_references,
    all_predictions,
    weights=[1.0, 0, 0, 0]
)

bleu_2 = corpus_bleu(
    all_references,
    all_predictions,
    weights=[0.5, 0.5, 0, 0]
)

bleu_3 = corpus_bleu(
    all_references,
    all_predictions,
    weights=[1/3, 1/3, 1/3, 0]
)

bleu_4 = corpus_bleu(
    all_references,
    all_predictions,
    weights=[0.25, 0.25, 0.25, 0.25]
)

print(f"BLEU-1: {bleu_1:.4f}")
print(f"BLEU-2: {bleu_2:.4f}")
print(f"BLEU-3: {bleu_3:.4f}")
print(f"BLEU-4: {bleu_4:.4f}")
