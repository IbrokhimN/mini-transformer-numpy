import numpy as np
from mini_transformer.config import Config
from mini_transformer.attention import mask, scaled_dot_product, attention, multi_head_attention

d_model = Config.d_model

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
