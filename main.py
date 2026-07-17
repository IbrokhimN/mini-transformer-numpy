# кароч ща все это дело собирать будем

import numpy as np
from mini_transformer.config import Config
from mini_transformer.tokenizer import encode, decode, unique_list
from mini_transformer.embeddings import get_embeddings
from mini_transformer.model import forward_model

# инициализируем веса модели 
def init_weights():
    vocab_size = Config.vocab_size
    d_model = Config.d_model
    d_ff = Config.d_ff
    n_layers = Config.n_layers
    n_heads = Config.n_heads

    blocks_weights = []
    for _ in range(n_layers):
        #создаем случайные веса для каждого слоя трансформера
        layer_weights = {
            'W_Q': np.random.randn(d_model, d_model) * 0.02,
            'W_K': np.random.randn(d_model, d_model) * 0.02,
            'W_V': np.random.randn(d_model, d_model) * 0.02,
            'W_O': np.random.randn(d_model, d_model) * 0.02,
            'W1': np.random.randn(d_model, d_ff) * 0.02,
            'W2': np.random.randn(d_ff, d_model) * 0.02,
            'b1': np.zeros(d_ff),
            'b2': np.zeros(d_model),
            'gamma1': np.ones(d_model),
            'beta1': np.zeros(d_model),
            'gamma2': np.ones(d_model),
            'beta2': np.zeros(d_model),
            'n_heads': n_heads
        }
        blocks_weights.append(layer_weights)

    W_out = np.random.randn(d_model, vocab_size) * 0.02
    gamma_final = np.ones(d_model)
    beta_final = np.zeros(d_model)

    return blocks_weights, W_out, gamma_final, beta_final

def softmax(x):
    # софтмаксим по последней оси для вероятностей
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def pipeline(input_text, blocks_weights, W_out, gamma_final, beta_final):
    # уххъ ща будет кайф
    tokens = encode(input_text)
    
    if len(tokens) > Config.max_seq_len:
        tokens = tokens[:Config.max_seq_len]
        
    x = get_embeddings(tokens)
    logits = forward_model(x, blocks_weights, W_out, gamma_final, beta_final)
    
    last_token_logits = logits[-1, :]
    probs = softmax(last_token_logits)
    
    next_token_idx = np.argmax(probs)
    next_char = decode([next_token_idx])
    
    return logits, probs, next_char

if __name__ == "__main__":
    with open("data/text.txt", "r", encoding="utf-8") as f:
        raw_text = f.read() 
    
    print("все загрузилось наверное, длинна текста ", len(raw_text))

    #тут мб возникнет вопрос а зачем я снова это посчитал? тут дело в том а что если текст поменять чуть чуть, тоесть у меня в конфиге
    #щас стоят данные из начального текста, а если я поменяю текст то уникальные символы могут чуток измениться и тогда надо будет обновить 
    # vocab_size в конфиге, а если щас просто сказать unique_chars = Config.unique_chars то будет фигня потому что в конфиге это значение не обновится
    # а значит и vocab_size тоже не обновится и будет ошибка
    unique_chars = sorted(list(set(raw_text)))
    Config.vocab_size = len(unique_chars)
    print("уникальных символов в тексте ", unique_chars)

    #создаем веса
    blocks_weights, W_out, gamma_final, beta_final = init_weights()
    
    # тут я просто беру случайный кусок текста из исходного текста чтоб протестировать пайплайн
    start_idx = np.random.randint(0, len(raw_text) - Config.max_seq_len)
    test_text = raw_text[start_idx : start_idx + Config.max_seq_len]

    #все это дело пропускаем через пайплайн
    logits, probs, next_char = pipeline(test_text, blocks_weights, W_out, gamma_final, beta_final)

    # ну все, финалочка
    print("тестовый текст--------------------- ", test_text)
    print("логиты последнего токена----------- ", logits[-1, :])
    print("вероятности последнего токена------ ", probs)
    print("предсказанный следующий символ----- ", next_char)