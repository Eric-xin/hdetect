import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, XLMRobertaTokenizer, MT5Tokenizer, AdamW
import argparse
from tqdm import tqdm

from dataloader import OLIDDataset
from model import MultiEncoderClassifier

def train_epoch(model, dataloader, optimizer, device, bert_tokenizer, xlmr_tokenizer, mt5_tokenizer):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    for batch in tqdm(dataloader, desc="Training"):
        texts = batch['text']
        labels_a = batch['label_a'].to(device)
        
        bert_enc = bert_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        xlmr_enc = xlmr_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        mt5_enc  = mt5_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        
        input_ids_bert = bert_enc['input_ids'].to(device)
        attention_mask_bert = bert_enc['attention_mask'].to(device)
        
        input_ids_xlmr = xlmr_enc['input_ids'].to(device)
        attention_mask_xlmr = xlmr_enc['attention_mask'].to(device)
        
        input_ids_mt5 = mt5_enc['input_ids'].to(device)
        attention_mask_mt5 = mt5_enc['attention_mask'].to(device)
        
        optimizer.zero_grad()
        logits_a, _ = model(input_ids_bert, attention_mask_bert,
                            input_ids_xlmr, attention_mask_xlmr,
                            input_ids_mt5, attention_mask_mt5)
        loss_a = criterion(logits_a, labels_a)
        loss = loss_a
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    mt5_tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
    
    train_dataset = OLIDDataset(args.train_file, tokenizer=bert_tokenizer, max_length=128)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    model = MultiEncoderClassifier(num_labels_a=2, num_labels_b=2)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, bert_tokenizer, xlmr_tokenizer, mt5_tokenizer)
    
    torch.save(model.state_dict(), args.output_model)
    print(f"Model saved to {args.output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="olid-training-v1.tsv", help="Path to the OLID training TSV file")
    parser.add_argument("--output_model", type=str, default="multiencoder_model.pt", help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()
    
    main(args)