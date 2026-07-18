import numpy as np
from mini_transformer.config import Config

d_model = Config.d_model

def mask(seq_len):
    #создаем маску для последовательности, чтобы оно не видело будущего
    m = np.triu(np.ones((seq_len, seq_len)), k=1)
    return m * -1e9

def scaled_dot_product(Q, K, V, d_k):
    #это ядро attention, принимает уже готовые Q, K, V и считает результат
    #d_k нужен отдельно, потому что для одной головы это head_dim, а не весь d_model
    seq_len = Q.shape[0]
    #формируем матрицу внимания
    scores = Q @ K.T / np.sqrt(d_k) # sqrt чтоб не было слишком больших значений
                                     # которые в последствии могут привести к проблемам с градиентами
    scores = scores + mask(seq_len) # добавляем маску чтобы не видеть будущего
    #softmax по строкам
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #делим на сумму по строкам чтоб получить вероятности
    #умножаем на V
    output = attention_weights @ V
    return output, attention_weights

def attention(x, W_Q, W_K, W_V):
    #x это результат с прошлого этапа эмбеддинги токенов + позиций форма seq_len, d_model
    #однa голова на весь дмодел, используем ядро с дк = дмодель
    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V
    return scaled_dot_product(Q, K, V, d_k=d_model)

def multi_head_attention(x, W_Q, W_K, W_V, W_O, n_heads):
    seq_len = x.shape[0]
    head_dim = d_model // n_heads
    #разделяем на головки
    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V
    Q = Q.reshape(seq_len, n_heads, head_dim)
    K = K.reshape(seq_len, n_heads, head_dim)
    V = V.reshape(seq_len, n_heads, head_dim)
    #подсчет для каждой головки
    outputs = []
    for i in range(n_heads):
        output, _ = scaled_dot_product(Q[:, i, :], K[:, i, :], V[:, i, :], d_k=head_dim)
        outputs.append(output)
    concatenated_output = np.concatenate(outputs, axis=-1) # обратно [seq_len, d_model]
    #финальная проекция, смешивает информацию со всех голов вместе
    final_output = concatenated_output @ W_O
    return final_output

# небольшие тесты
#from embeddings import get_embeddings
#indices = [28, 25, 32, 32, 35]
#x = get_embeddings(indices)

#W_Q = np.random.randn(d_model, d_model) * 0.02
#W_K = np.random.randn(d_model, d_model) * 0.02
#W_V = np.random.randn(d_model, d_model) * 0.02
#W_O = np.random.randn(d_model, d_model) * 0.02

#output = multi_head_attention(x, W_Q, W_K, W_V, W_O, n_heads=Config.n_heads)
#print(output.shape)  # (5, 32)

# сикс севен
# «Мечта может стать поддержкой или источником страданий. 
# Мечта может наполнить человека жизнью или убить его.
# Даже если мечта оставила человека, её частица будет всегда тлеть в глубине его сердца. 
# Каждый должен хоть раз представить, что его жизнь - это жизнь божьего мученика, преследующего свою мечту.» Гриффи
