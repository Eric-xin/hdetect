import torch
from torch.utils.data import Dataset
import pandas as pd

class OLIDDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        """
        Args:
            file_path (str): Path to the TSV file containing the OLID dataset.
            tokenizer: Pre-trained tokenizer to tokenize the tweet text.
            max_length (int): Maximum sequence length for tokenization.
        """
        # Read the TSV file (tab-separated)
        self.data = pd.read_csv(file_path, sep='\t')
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Define label mapping for Task A only (offensive language identification)
        self.label_map_a = {"NOT": 0, "OFF": 1}
        
        # Filter out rows with missing labels for Task A
        self.data = self.data[(self.data['subtask_a'].notna()) & (self.data['subtask_a'] != "NULL")]

    def __len__(self):
        return len(self.data)
    
    def process_label(self, label, mapping):
        return mapping[label]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tweet = row['tweet']
        label_a = self.process_label(row['subtask_a'], self.label_map_a)
        
        # Tokenize the tweet text
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()          # Tensor shape: [max_length]
        attention_mask = encoding['attention_mask'].squeeze()  # Tensor shape: [max_length]
        
        return {
            'text': tweet,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_a': label_a
        }