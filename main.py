from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
import os
import warnings

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

app = Flask(__name__)

class Seq2SeqTokenizer:
    def __init__(self, word2idx=None):
        if word2idx is None:
            word2idx = {}
        self.word2idx = word2idx
        self.idx2word = {idx: word for word, idx in word2idx.items()}
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3
        }
    
    def get_vocab_size(self):
        return len(self.word2idx)
    
    def encode(self, text, max_length=64):
        words = text.lower().split()
        tokens = ['[CLS]'] + words + ['[SEP]']
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + ['[SEP]']
        
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx.get('[UNK]', 1))
        
        if len(indices) < max_length:
            indices = indices + [self.word2idx.get('[PAD]', 0)] * (max_length - len(indices))
        
        attention_mask = [1 if idx != self.word2idx.get('[PAD]', 0) else 0 for idx in indices]
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class IndependentSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, max_len=64):
        super().__init__()
        self.d_model = d_model
        
        # Encoder
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_pos = PositionalEncoding(d_model, max_len)
        self.encoder_dropout = nn.Dropout(0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_pos = PositionalEncoding(d_model, max_len)
        self.decoder_dropout = nn.Dropout(0.1)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_input_ids, src_key_padding_mask, tgt_input_ids, tgt_key_padding_mask):
        # encoder
        src_emb = self.encoder_embedding(src_input_ids) * math.sqrt(self.d_model)
        src_emb = self.encoder_pos(src_emb)
        src_emb = self.encoder_dropout(src_emb)
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # decoder
        tgt_emb = self.decoder_embedding(tgt_input_ids) * math.sqrt(self.d_model)
        tgt_emb = self.decoder_pos(tgt_emb)
        tgt_emb = self.decoder_dropout(tgt_emb)
        
        tgt_len = tgt_input_ids.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        causal_mask = causal_mask.to(tgt_input_ids.device)
        
        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.output_proj(output)


class Seq2SeqCorrector:
    def __init__(self, model_path='independent_seq2seq_corrector.pth', tokenizer_path='seq2seq_tokenizer.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device: {self.device}")
        
        # загрузка модели и токенизатора
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_path, tokenizer_path)
        if self.model and self.tokenizer:
            self.model.eval()
            print("seq2seq model and tokenizer loaded successfully")
        else:
            print("Failed to load model or tokenizer")
    
    def load_model_and_tokenizer(self, model_path, tokenizer_path):
        try:
            # загрузка токенизатора
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                print(f"Tokenizer loaded with {tokenizer.get_vocab_size()} tokens")
            else:
                print(f"Tokenizer file {tokenizer_path} not found!")
                return None, None
            
            # загрузка модели
            state_dict = torch.load(model_path, map_location=self.device)
            vocab_size = tokenizer.get_vocab_size()
            
            model = IndependentSeq2SeqModel(
                vocab_size=vocab_size,
                d_model=256,
                num_layers=6
            )
            
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading model and tokenizer: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def correct_sentence(self, sentence, max_len=64):
        if not self.model or not self.tokenizer:
            return "[Модель не загружена]"
        
        try:
            with torch.no_grad():
                # кодируем входное предложение
                src_enc = self.tokenizer.encode(sentence, max_len)
                src_ids = src_enc['input_ids'].to(self.device)
                src_mask = src_enc['attention_mask'].to(self.device)
                
                # булевые маски
                src_key_padding_mask = (src_mask == 0)
                
                # жадная генерация
                generated = [self.tokenizer.word2idx.get('[CLS]', 2)]
                repeated_count = 0
                last_token = None
                
                for _ in range(max_len - 1):
                    tgt_tensor = torch.tensor([generated], dtype=torch.long).to(self.device)
                    # для decoder: все позиции активны (нет padding)
                    tgt_key_padding_mask = torch.zeros(len(generated), dtype=torch.bool).to(self.device)
                    
                    # размерность батча
                    logits = self.model(
                        src_ids.unsqueeze(0), 
                        src_key_padding_mask.unsqueeze(0),
                        tgt_tensor, 
                        tgt_key_padding_mask.unsqueeze(0)
                    )
                    next_token = logits[0, -1].argmax().item()
                    
                    # предотвращение повторений
                    if next_token == last_token:
                        repeated_count += 1
                        if repeated_count > 3:
                            break
                    else:
                        repeated_count = 0
                        last_token = next_token
                    
                    generated.append(next_token)
                    
                    if next_token == self.tokenizer.word2idx.get('[SEP]', 3):
                        break
                
                # декодирование
                tokens = []
                for idx in generated[1:-1]:  # пропускаем [CLS] и [SEP]
                    if idx in self.tokenizer.idx2word:
                        tokens.append(self.tokenizer.idx2word[idx])
                    else:
                        tokens.append('[UNK]')
                
                result = ' '.join(tokens)
                
                # восстановление регистра первого слова
                if sentence and sentence[0].isupper():
                    result = result.capitalize()
                
                return result
                
        except Exception as e:
            print(f"Correction error: {e}")
            import traceback
            traceback.print_exc()
            return sentence

# инициализация корректора
corrector = Seq2SeqCorrector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    sentence = request.json.get('sentence', '').strip()
    
    if not sentence:
        return jsonify({'error': 'no sentence provided'})
    
    result = detector.detect_error(sentence)
    
    response = {
        'sentence': result['sentence'],
        'has_error': result['has_error'],
        'confidence': result['confidence'],
        'similarity': result['similarity'],
        'category': result['category'],
        'most_similar': result['most_similar'],
        'top_similar': result['top_similar'],
        'model_type': result['model_type'],
        'details': {
            'avg_similarity_correct': result.get('avg_similarity_correct', 0),
            'avg_similarity_incorrect': result.get('avg_similarity_incorrect', 0)
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':

    print("детекция семантических ошибок в английском языке")
    print(f"device: {detector.device}")
    print(f"vocabulary size: {len(detector.tokenizer.word2idx)}")
    print("running on: http://localhost:5000")
    
    # тест модели
    test_cases = [
        "The teenager does tour at most nine restaurants.",
        "No teenager does tour at most nine restaurants.",
        "No teacher assigned more than five tasks.",
        "Every senator notices at most four windows.",
        "No senator notices at most four windows."
    ]
    
    for test in test_cases:
        result = detector.detect_error(test)
        status = "correct" if (test in detector.correct_sentences) != result['has_error'] else "error"
        print(f"\n{status} - '{test}'")
        print(f"predict: {'ERROR' if result['has_error'] else 'OK'} (conf: {result['confidence']:.2f})")
        print(f"category: {result['category']}")
        print(f"most similar: {result['most_similar'][:60]}...")
    
    app.run(debug=True, port=5000)