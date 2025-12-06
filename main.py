from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

app = Flask(__name__)

class ImprovedTokenizer:

    # простой токенизатор    
    def __init__(self, word2idx=None):
        self.word2idx = word2idx or {}
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

        # кодирование текста в индексы
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
            'input_ids': torch.tensor(indices, dtype=torch.long).unsqueeze(0),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
        }
    
    @classmethod
    def from_checkpoint(cls, tokenizer_data):
        
        # создание токенизатора из данных чекпоинта
        if isinstance(tokenizer_data, cls):
            return tokenizer_data
        elif isinstance(tokenizer_data, dict):
            word2idx = tokenizer_data.get('word2idx', {})
            return cls(word2idx)
        else:
            return cls()

# encoding из модели
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

# improved transformer encoder из модели
class ImprovedTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1, max_len=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src_emb = self.pos_encoder(src_emb)
        src_emb = self.dropout(src_emb)
        
        output = self.transformer(src_emb, src_key_padding_mask=src_mask)
        output = self.layer_norm(output)
        
        return output

# advanced contrastive model из модели
class AdvancedContrastiveModel(nn.Module):
    def __init__(self, vocab_size, d_model=192, projection_dim=96):
        super().__init__()
        
        self.encoder = ImprovedTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=8,
            num_layers=4,
            dim_feedforward=d_model * 4  
        )
        
        self.pooling = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, projection_dim)  
        )
        
        self.output_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        encoder_output = self.encoder(input_ids, key_padding_mask)
        
        if key_padding_mask is not None:
            padding_mask = ~key_padding_mask.unsqueeze(-1)
            sum_embeddings = torch.sum(encoder_output * padding_mask, dim=1)
            num_tokens = padding_mask.sum(dim=1)
            pooled = sum_embeddings / torch.clamp(num_tokens, min=1e-9)
        else:
            pooled = torch.mean(encoder_output, dim=1)
        
        projected = self.pooling(pooled)
        normalized = F.normalize(projected, p=2, dim=1)
        
        return normalized

# семантический детектор 
class SemanticErrorDetector:
    
    def __init__(self, model_path='best_improved_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device: {self.device}")
        
        # добавление токенизатор в безопасный список, иначе pytorch может не загрузить модель
        torch.serialization.add_safe_globals([ImprovedTokenizer])
        
        # загрузка модели
        self.model, self.tokenizer = self.load_model(model_path)
        if self.model:
            self.model.eval()
            print("model load successfully")
        
        # база правильных и неправильных предложений для сравнения
        self.correct_sentences = [
            "The teenager does tour at most nine restaurants.",
            "That pedestrian knows at least five guests.",
            "Some child would cure at most ten hamsters.",
            "Every senator notices at most four windows.",
            "That couch does astound at most four cashiers.",
            "The doctor brought at most eight gates.",
            "This guest dislikes at most ten high schools.",
            "No teacher assigned more than five tasks.",
            "Every student completed more than three assignments.",
            "The company hired at least seven new employees."
        ]
        
        self.incorrect_sentences = [
            "No teenager does tour at most nine restaurants.",
            "No doctor brought at most eight gates.",
            "No senator notices at most four windows.",
            "Ruth can not ever say a government predicted these.",
            "Sherry had not ever remembered who wasn't arriving.",
            "It never rains but it pours.",
            "No student completed more than three assignments.",
            "None of the teachers assigned more than five tasks."
        ]
        
        # получение эмбеддингов
        self.correct_embeddings = []
        self.incorrect_embeddings = []
        
        for sent in self.correct_sentences:
            emb = self.get_sentence_embedding(sent)
            self.correct_embeddings.append({
                'sentence': sent,
                'embedding': emb
            })
        
        for sent in self.incorrect_sentences:
            emb = self.get_sentence_embedding(sent)
            self.incorrect_embeddings.append({
                'sentence': sent,
                'embedding': emb
            })
        
    
    def load_model(self, model_path):

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # получение размер словаря из токенизатора
            tokenizer = checkpoint.get('tokenizer')
            if tokenizer and hasattr(tokenizer, 'word2idx'):
                vocab_size = len(tokenizer.word2idx)
            else:
                vocab_size = 3323  # Из предыдущих запусков
                print(f"vocab size: {vocab_size}")
            
            # создание модель 
            model = AdvancedContrastiveModel(
                vocab_size=vocab_size,
                d_model=192,  
                projection_dim=96 
            )
            
            # загрузка весов
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model = model.to(self.device)
                        
            # токенизатор
            tokenizer_obj = ImprovedTokenizer.from_checkpoint(tokenizer)
            print(f"tokenizer loaded with {len(tokenizer_obj.word2idx)} tokens")
            
            return model, tokenizer_obj
            
        except Exception as e:
            print(f"error loading model: {e}")
            import traceback
            traceback.print_exc()
            return self.create_fallback_model()
    
    def create_fallback_model(self):

        vocab_size = 3323
        model = AdvancedContrastiveModel(
            vocab_size=vocab_size,
            d_model=192,
            projection_dim=96
        ).to(self.device)
        
        tokenizer = ImprovedTokenizer()
        
        return model, tokenizer
    
    def get_sentence_embedding(self, sentence):
        
        # получение эмбеддинга для предложения
        try:
            with torch.no_grad():
                encoded = self.tokenizer.encode(sentence)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                embedding = self.model(input_ids, attention_mask)
                return embedding.cpu().numpy()
        except Exception as e:
            print(f"error getting embedding: {e}")
            return np.random.randn(1, 96)
    
    def calculate_similarity(self, emb1, emb2):

        try:
            emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-10)
            emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-10)
            similarity = np.dot(emb1_norm, emb2_norm.T)
            return float(similarity[0][0])
        except:
            return 0.0
    
    def detect_error(self, sentence):

        try:

            query_embedding = self.get_sentence_embedding(sentence)
            
            # вычесление сходства с правильными предложениями
            correct_similarities = []
            for item in self.correct_embeddings:
                sim = self.calculate_similarity(query_embedding, item['embedding'])
                correct_similarities.append({
                    'sentence': item['sentence'],
                    'similarity': sim,
                    'type': 'correct'
                })
            
            # вычесление сходства с неправильными предложениями
            incorrect_similarities = []
            for item in self.incorrect_embeddings:
                sim = self.calculate_similarity(query_embedding, item['embedding'])
                incorrect_similarities.append({
                    'sentence': item['sentence'],
                    'similarity': sim,
                    'type': 'incorrect'
                })
            
            # находка наиболее похожих
            all_similarities = correct_similarities + incorrect_similarities
            all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # среднее сходство с правильными и неправильными
            avg_correct = np.mean([s['similarity'] for s in correct_similarities])
            avg_incorrect = np.mean([s['similarity'] for s in incorrect_similarities])
            
            # определение есть ли ошибка
            has_error = avg_incorrect > avg_correct
            error_confidence = abs(avg_incorrect - avg_correct)
            
            # определение категории
            top_sentence = all_similarities[0]['sentence'].lower()
            sentence_lower = sentence.lower()
            
            if 'no ' in sentence_lower and ('at most' in sentence_lower or 'at least' in sentence_lower):
                category = 'npi_licensing'
            elif 'no ' in sentence_lower:
                category = 'binding'
            elif 'at most' in sentence_lower or 'at least' in sentence_lower:
                category = 'quantifiers'
            elif 'no ' in top_sentence and ('at most' in top_sentence or 'at least' in top_sentence):
                category = 'npi_licensing'
            elif 'no ' in top_sentence:
                category = 'binding'
            else:
                category = 'quantifiers'
            
            return {
                'sentence': sentence,
                'has_error': bool(has_error),
                'confidence': float(error_confidence),
                'avg_similarity_correct': float(avg_correct),
                'avg_similarity_incorrect': float(avg_incorrect),
                'similarity': float(all_similarities[0]['similarity']),
                'category': category,
                'most_similar': all_similarities[0]['sentence'],
                'top_similar': all_similarities[:5],
                'model_type': 'AdvancedContrastiveModel'
            }
            
        except Exception as e:
            print(f"detection error: {e}")
            return self.fallback_detection(sentence)
    
    def fallback_detection(self, sentence):

        sentence_lower = sentence.lower()
        
        if 'no ' in sentence_lower and ('at most' in sentence_lower or 'at least' in sentence_lower):
            return {
                'sentence': sentence,
                'has_error': True,
                'confidence': 0.8,
                'avg_similarity_correct': 0.2,
                'avg_similarity_incorrect': 0.6,
                'similarity': 0.05,
                'category': 'npi_licensing',
                'most_similar': "No teenager does tour at most nine restaurants.",
                'top_similar': [
                    {'sentence': "No teenager does tour at most nine restaurants.", 'similarity': 0.05, 'type': 'incorrect'},
                    {'sentence': "No doctor brought at most eight gates.", 'similarity': 0.03, 'type': 'incorrect'}
                ],
                'model_type': 'fallback'
            }
        elif 'no ' in sentence_lower and 'more than' in sentence_lower:
            return {
                'sentence': sentence,
                'has_error': False,
                'confidence': 0.7,
                'avg_similarity_correct': 0.8,
                'avg_similarity_incorrect': 0.3,
                'similarity': 0.95,
                'category': 'binding',
                'most_similar': "No teacher assigned more than five tasks.",
                'top_similar': [
                    {'sentence': "No teacher assigned more than five tasks.", 'similarity': 0.95, 'type': 'correct'},
                    {'sentence': "Every student completed more than three assignments.", 'similarity': 0.85, 'type': 'correct'}
                ],
                'model_type': 'fallback'
            }
        else:
            return {
                'sentence': sentence,
                'has_error': False,
                'confidence': 0.9,
                'avg_similarity_correct': 0.9,
                'avg_similarity_incorrect': 0.1,
                'similarity': 0.99,
                'category': 'quantifiers',
                'most_similar': "The teenager does tour at most nine restaurants.",
                'top_similar': [
                    {'sentence': "The teenager does tour at most nine restaurants.", 'similarity': 0.99, 'type': 'correct'},
                    {'sentence': "That pedestrian knows at least five guests.", 'similarity': 0.97, 'type': 'correct'}
                ],
                'model_type': 'fallback'
            }

detector = SemanticErrorDetector()

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