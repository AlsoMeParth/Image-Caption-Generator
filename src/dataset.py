import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor

def load_pairs(captions_file, image_dir):
    pairs = []
    df = pd.read_csv(captions_file)
    for _, rows in df.iterrows():
        image_path = Path(image_dir)/ rows["image"]
        caption = str(rows["caption"]).strip()
        if image_path.exists() and caption:
            pairs.append((image_path, caption))
    return pairs
        

def split_by_image(pairs, test_size = 0.1, val_size = 0.1, seed = 42):
    image_paths = sorted(set(path for path, _ in pairs))
    
    train_paths, remaining_paths = train_test_split(
        image_paths, 
        test_size = test_size+val_size,
        random_state=seed
    )

    val_paths, test_paths = train_test_split(
        remaining_paths,
        test_size = test_size / (val_size+test_size),
        random_state = seed
    )

    train_paths = set(train_paths)
    val_paths = set(val_paths)
    test_paths = set(test_paths)

    train_pairs = [(p, c) for p, c in pairs if p in train_paths]
    val_pairs = [(p, c) for p, c in pairs if p in val_paths]
    test_pairs = [(p, c) for p, c in pairs if p in test_paths]

    return train_pairs, val_pairs, test_pairs

def load_processor():
    image_processor = AutoImageProcessor.from_pretrained( 
        "google/vit-base-patch16-224-in21k"
        )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return image_processor, tokenizer


class Flickr8kDataset(Dataset):
    def __init__(self, pairs, image_processor, tokenizer, max_length = 32):
        self.pairs = pairs
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        image_path, caption = self.pairs[index]
        with Image.open(image_path) as image:
            pixel_values = self.image_processor(
                images = image.convert("RGB"),
                return_tensors="pt",
            )["pixel_values"].squeeze(0)

        caption = (
            self.tokenizer.eos_token
            + caption.strip()
            + self.tokenizer.eos_token
        )
        tokens = self.tokenizer(
            caption,
            max_length = self.max_length,
            truncation = True,
            padding = "max_length",
            return_tensors = "pt"
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return{
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

