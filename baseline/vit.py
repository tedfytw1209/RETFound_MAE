import os
import numpy as np
import pandas as pd
from PIL import Image
from evaluate import load
import torch
from torch.utils.data import Dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer

TRAIN_CSV = 'CellData/OCT/DME_train.csv'
VAL_CSV = 'CellData/OCT/DME_test.csv'
IMAGE_FOLDER = '.'
PRETRAINED_MODEL = 'google/vit-base-patch16-224-in21k'
NUM_CLASSES = 2
OUTPUT_DIR = 'output'

class OCTDataset(Dataset):
    def __init__(self, csv_file, image_folder, processor):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data['image'].iloc[idx])
        label = int(self.data['label'][idx]<=0)
        
        image = Image.open(img_name).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        
        # Hugging Face Trainer expects dict-like output
        return {k: v.squeeze(0) for k, v in inputs.items()}

processor = ViTImageProcessor.from_pretrained(PRETRAINED_MODEL)
train_dataset = OCTDataset(TRAIN_CSV, IMAGE_FOLDER, processor)
val_dataset = OCTDataset(VAL_CSV, IMAGE_FOLDER, processor)

model = ViTForImageClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=NUM_CLASSES,
    id2label={0: "Negative", 1: "Positive"},
    label2id={"Negative": 0, "Positive": 1}
)

auc_metric = load("roc_auc")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return auc_metric.compute(prediction_scores=probs, references=labels)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir=f'{OUTPUT_DIR}/logs',
    logging_steps=50,
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="roc_auc",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(f"{OUTPUT_DIR}/vit_DME")
processor.save_pretrained(f"{OUTPUT_DIR}/vit_DME")

eval_results = trainer.evaluate()
print(f"Validation AUC: {eval_results['eval_roc_auc']:.4f}")