import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments


data = pd.read_csv(r'botGPT/custom_model/qa_data.csv') 

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare the dataset for training
max_length = 512 
train_encodings = tokenizer(data['question'].tolist(), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
train_labels = tokenizer(data['answer'].tolist(), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')


class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels['input_ids'][idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

train_dataset = QADataset(train_encodings, train_labels)

# training arguments
training_args = TrainingArguments(
    output_dir='botGPT/custom_data',    
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='botGPT/custom_data/logs', 
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune model
trainer.train()

# Save the model
model.save_pretrained('botGPT/custom_data')
tokenizer.save_pretrained('botGPT/custom_data')