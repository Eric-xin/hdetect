import torch
import torch.nn as nn
from transformers import BertModel, XLMRobertaModel, MT5ForConditionalGeneration

class MultiEncoderClassifier(nn.Module):
    def __init__(self, num_labels_a=2, num_labels_b=2, dropout_prob=0.1):
        """
        Multi-encoder classifier that fuses representations from:
         - BERT (multilingual cased)
         - XLM-RoBERTa (base)
         - mT5 (using its encoder)
        and applies two separate classification heads for Subtask A and Subtask B.
        """
        super(MultiEncoderClassifier, self).__init__()
        
        # Load pre-trained models
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.xlmr = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.mt5 = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
        
        # Hidden dimensions from each model
        bert_hidden = self.bert.config.hidden_size          # typically 768
        xlmr_hidden = self.xlmr.config.hidden_size          # typically 768
        mt5_hidden  = self.mt5.config.d_model               # typically 512
        
        # Combined representation size
        combined_size = bert_hidden + xlmr_hidden + mt5_hidden
        
        # Classification head for Subtask A (e.g., NOT vs OFF)
        self.classifier_a = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(combined_size, combined_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(combined_size // 2, num_labels_a)
        )
        
        # Classification head for Subtask B (e.g., UNT vs TIN)
        self.classifier_b = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(combined_size, combined_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(combined_size // 2, num_labels_b)
        )
        
    def forward(self, input_ids_bert, attention_mask_bert,
                input_ids_xlmr, attention_mask_xlmr,
                input_ids_mt5, attention_mask_mt5):
        # BERT encoding
        bert_outputs = self.bert(input_ids=input_ids_bert, attention_mask=attention_mask_bert)
        bert_cls = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # XLM-RoBERTa encoding
        xlmr_outputs = self.xlmr(input_ids=input_ids_xlmr, attention_mask=attention_mask_xlmr)
        xlmr_cls = xlmr_outputs.last_hidden_state[:, 0, :]  # first token
        
        # mT5 encoding (using encoder outputs)
        mt5_encoder_outputs = self.mt5.encoder(input_ids=input_ids_mt5, attention_mask=attention_mask_mt5)
        mt5_cls = mt5_encoder_outputs.last_hidden_state[:, 0, :]  # simple pooling strategy
        
        # Concatenate the representations
        combined = torch.cat([bert_cls, xlmr_cls, mt5_cls], dim=1)
        
        # Get logits for each subtask
        logits_a = self.classifier_a(combined)
        logits_b = self.classifier_b(combined)
        return logits_a, logits_b