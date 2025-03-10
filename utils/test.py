import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, XLMRobertaTokenizer, MT5Tokenizer
import argparse
from tqdm import tqdm

from dataloader import OLIDDataset
from model import MultiEncoderClassifier

def evaluate(model, dataloader, device, bert_tokenizer, xlmr_tokenizer, mt5_tokenizer):
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    correct_a = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
            
            logits_a, _ = model(input_ids_bert, attention_mask_bert,
                                input_ids_xlmr, attention_mask_xlmr,
                                input_ids_mt5, attention_mask_mt5)
            loss_a = criterion(logits_a, labels_a)
            total_loss += loss_a.item()
            
            preds_a = torch.argmax(logits_a, dim=1)
            correct_a += (preds_a == labels_a).sum().item()
            total += labels_a.size(0)
    
    avg_loss = total_loss / len(dataloader)
    acc_a = correct_a / total if total > 0 else 0
    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Task A Accuracy: {acc_a*100:.2f}%")
    return avg_loss, acc_a

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizers
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    mt5_tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
    
    test_dataset = OLIDDataset(args.test_file, tokenizer=bert_tokenizer, max_length=128)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = MultiEncoderClassifier(num_labels_a=2, num_labels_b=2)
    model.load_state_dict(torch.load(args.input_model, map_location=device))
    model.to(device)
    
    evaluate(model, test_loader, device, bert_tokenizer, xlmr_tokenizer, mt5_tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="olid-test.tsv", help="Path to the OLID test TSV file")
    parser.add_argument("--input_model", type=str, default="multiencoder_model.pt", help="Path to the trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    args = parser.parse_args()
    
    main(args)