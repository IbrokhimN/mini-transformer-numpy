import numpy as np
from mini_transformer.config import Config

d_model = Config.d_model

def mask(seq_len):
    #это крч для того что бы он не смотрел на будущие токены
    m = np.triu(np.ones((seq_len, seq_len)), k=1)
    return m * -1e9

def scaled_dot_product(Q, K, V, d_k):
    #тут мы считаем внимание, и крч принимаем Q K V и рез выдаем
    seq_len = Q.shape[0]
    scores = Q @ K.T / np.sqrt(d_k)
    scores = scores + mask(seq_len)
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    output = attention_weights @ V
    return output, attention_weights

def attention(x, W_Q, W_K, W_V):
    # эт база этеншена
    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V
    return scaled_dot_product(Q, K, V, d_k=d_model)

def multi_head_attention(x, W_Q, W_K, W_V, W_O, n_heads):
    # делаем несколько головок внимания, каждая голова видит разные аспекты входа
    seq_len = x.shape[0]
    head_dim = d_model // n_heads

    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V
    # решейпим Q K V для каждой головы, теперь у нас форма [seq_len, n_heads, head_dim]
    Q = Q.reshape(seq_len, n_heads, head_dim)
    K = K.reshape(seq_len, n_heads, head_dim)
    V = V.reshape(seq_len, n_heads, head_dim)

    outputs = []
    for i in range(n_heads):
        output, _ = scaled_dot_product(Q[:, i, :], K[:, i, :], V[:, i, :], d_k=head_dim)
        outputs.append(output)
    
    # объединяем все головы обратно в форму
    concatenated_output = np.concatenate(outputs, axis=-1)
    final_output = concatenated_output @ W_O
    return final_output

def layer_norm(x, gamma, beta, eps=1e-6):
    # надо найти среднее и дисперсию по последней оси потом нормализовать и умножить на гамму и прибавить бета
    # потому что без гаммы и беты нейросеть не сможет обучаться потому что нормализация убирает смещение и масштаб
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    normalized_x = (x - mean) / np.sqrt(variance + eps)
    
    return gamma * normalized_x + beta

def res_attention_block(x, W_Q, W_K, W_V, W_O, n_heads, gamma1, beta1):
    # это нам нужно для того что бы вход был нормализованным это помогает обучению
    # тут тоже нормализация входа и потом внимание и потом прибавка от внимания
    normed_x = layer_norm(x, gamma1, beta1)
    attention_output = multi_head_attention(normed_x, W_Q, W_K, W_V, W_O, n_heads)
    return x + attention_output

def feed_forward_block(x, W1, b1, W2, b2, gamma2, beta2):
    # ну тут тоже нормализация входа и потом feed forward и потом прибавка от feed forward
    # только feed forward это просто два линейных слоя с активацией между ними
    normed_x = layer_norm(x, gamma2, beta2)
    ff_output = np.maximum(0, normed_x @ W1 + b1) @ W2 + b2
    return x + ff_output