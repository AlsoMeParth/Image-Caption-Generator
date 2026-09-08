import os
if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
from src.model import ImageCaptioningModel
from src.dataset import Flickr8kDataset, load_pairs, split_by_image, load_processor
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm

TEST_SIZE = 0.1
VAL_SIZE = 0.1
SEED = 42
NUM_EPOCHS = 20
PATIENCE = 3
MIN_DELTA = 1e-4
GRAD_ACCUMULATION_STEPS = 4
BATCH_SIZE = 2
NUM_WORKERS = 0
LR = 1e-4
WEIGHT_DECAY = 0.01
MAX_NORM = 1.0
CAPTION_PATH = "../Flickr8k/captions.txt"
IMAGE_PATH = "../Flickr8k/Images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Detected Device: {torch.cuda.get_device_name() if device.type=='cuda' else 'cpu'}")
@torch.inference_mode()
def evaluate(model, val_loader, device):
    model.eval()

    running_val_loss = 0.0

    progress_bar_val = tqdm(
        val_loader,
        desc = f"Epoch {epoch+1}/{NUM_EPOCHS}"  #"Validation"
    )

    for batch in progress_bar_val:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }
        with torch.amp.autocast(
            device_type = device.type,
            enabled = (device.type=="cuda")
        ):
            outputs = model(**batch)
            loss = outputs.loss

        running_val_loss += loss.item()

        progress_bar_val.set_postfix(
            loss=f"{loss.item():.4f}"
        )
    return running_val_loss/len(val_loader)
print("LOADING MODEL: ")
model = ImageCaptioningModel()
model.freeze_for_alignment()

trainable_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
print("Model Loaded Successfully")
print(f"Trainable parameters: {trainable_params:,}")

print(f"\n\nGetting Captions & Images...")
pairs = load_pairs(CAPTION_PATH, IMAGE_PATH)
train_pairs, val_pairs, test_pairs = split_by_image(pairs, TEST_SIZE, VAL_SIZE, SEED)
image_processor, tokenizer = load_processor()

train_dataset = Flickr8kDataset(train_pairs, image_processor, tokenizer)
val_dataset = Flickr8kDataset(val_pairs, image_processor, tokenizer)


print(f"Getting DataLoaders Ready...")
train_loader = DataLoader(
    train_dataset,
    batch_size =BATCH_SIZE,
    shuffle = True,
    num_workers = NUM_WORKERS,
    pin_memory = (device.type == "cuda")
)
#val loader
val_loader = DataLoader(
    val_dataset,
    batch_size =BATCH_SIZE,
    shuffle = False,
    num_workers = NUM_WORKERS,
    pin_memory = (device.type == "cuda")
)

model = model.to(device)

trainable_parameters = [
    parameter 
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr = LR,
    weight_decay = WEIGHT_DECAY
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled = (device.type == "cuda")
)
print(f"Done!")
print(f"\n\n-------TRAINING-------")

best_val_loss = float('inf')
epochs_without_improvement = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    running_train_loss = 0.0
    optimizer.zero_grad(set_to_none = True)

    progress_bar_train = tqdm(
        train_loader,
        desc = f"Epoch {epoch+1}/{NUM_EPOCHS}"
    )
  
    for step, batch in enumerate(progress_bar_train):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }
        with torch.amp.autocast(
            device_type = device.type,
            enabled = (device.type=="cuda")
        ):
            outputs = model(**batch)
            raw_train_loss = outputs.loss
            
            loss = raw_train_loss/GRAD_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        should_update=(
            (step+1)%GRAD_ACCUMULATION_STEPS==0
            or (step+1) == len(train_loader)
        )

        if should_update:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm = MAX_NORM
            )
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
        
        running_train_loss += raw_train_loss.item()

        progress_bar_train.set_postfix(
            loss=f"{raw_train_loss.item():.4f}"
        )

    val_loss = evaluate(model, val_loader, device)
    
    print(
        f"Epoch {epoch+1} | "
        f"Training loss: {running_train_loss/len(train_loader):.4f} | Val_loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            },
            "best_model.pt"
        )

        print("Validation loss improved — saved new best model.")

    else:
        epochs_without_improvement += 1

        print(
            f"No validation improvement for "
            f"{epochs_without_improvement}/{PATIENCE} epoch(s)."
        )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break
