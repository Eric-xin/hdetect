import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, XLMRobertaTokenizer, MT5Tokenizer, AdamW
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm

# Import your custom modules (adjust the paths as necessary)
# from utils.dataloader import OLIDDataset
from utils import OLIDDataset
from model import MultiEncoderClassifier

def train_epoch(model, dataloader, optimizer, device, 
                bert_tokenizer, xlmr_tokenizer, mt5_tokenizer, max_length):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    for batch in tqdm(dataloader, desc="Training"):
        texts = batch['text']
        # Only use Task A labels; ignore Task B
        labels_a = batch['label_a'].to(device)
        
        # Tokenize texts using each model's tokenizer
        bert_enc = bert_tokenizer(texts, padding=True, truncation=True, 
                                  return_tensors='pt', max_length=max_length)
        xlmr_enc = xlmr_tokenizer(texts, padding=True, truncation=True, 
                                  return_tensors='pt', max_length=max_length)
        mt5_enc  = mt5_tokenizer(texts, padding=True, truncation=True, 
                                  return_tensors='pt', max_length=max_length)
        
        input_ids_bert = bert_enc['input_ids'].to(device)
        attention_mask_bert = bert_enc['attention_mask'].to(device)
        
        input_ids_xlmr = xlmr_enc['input_ids'].to(device)
        attention_mask_xlmr = xlmr_enc['attention_mask'].to(device)
        
        input_ids_mt5 = mt5_enc['input_ids'].to(device)
        attention_mask_mt5 = mt5_enc['attention_mask'].to(device)
        
        optimizer.zero_grad()
        # Get outputs; we ignore the second output (for Task B)
        logits_a = model(input_ids_bert, attention_mask_bert,
                            input_ids_xlmr, attention_mask_xlmr,
                            input_ids_mt5, attention_mask_mt5)
        loss_a = criterion(logits_a, labels_a)
        loss = loss_a
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main(args):
    # Automatically select CUDA if available, else MPS if available, else CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizers
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    mt5_tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
    
    # Create training dataset and dataloader (dataloader should filter for valid Task A labels)
    train_dataset = OLIDDataset(args.train_file, tokenizer=bert_tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model (we still load two heads, but we only use Task A head here)
    model = MultiEncoderClassifier(num_labels_a=2)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=args.log_dir)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, 
                                 bert_tokenizer, xlmr_tokenizer, mt5_tokenizer, args.max_length)
        print(f"Training Loss: {train_loss:.4f}")
        # Log the training loss for the epoch
        writer.add_scalar('Loss/train', train_loss, epoch+1)
    
    torch.save(model.state_dict(), args.output_model)
    print(f"Model saved to {args.output_model}")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="olid-training-v1.tsv", help="Path to the OLID training TSV file")
    parser.add_argument("--output_model", type=str, default="multiencoder_model.pt", help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum token length")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save TensorBoard logs")
    args = parser.parse_args()
    
    main(args)