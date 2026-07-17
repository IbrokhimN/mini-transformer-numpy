import numpy as np
from mini_transformer.block import transformer_block
from mini_transformer.layers import layer_norm

def transformer_stack(x, blocks_weights):
    """
        кароч как это работает?:
            x это входные эмбеддинги токенов + позиций, форма [seq_len, d_model]
            blocks_weights это список словарей с весами для каждого блока трансформера
            каждый словарь содержит W_Q, W_K, W_V, W_O, n_heads, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2
            мы проходим по каждому блоку и применяем transformer_block к x, обновляя его
            в конце возвращаем x, который теперь содержит выход после всех блоков трансформера
    """
    for weights in blocks_weights:
        x = transformer_block(
            x,
            W_Q=weights['W_Q'],
            W_K=weights['W_K'],
            W_V=weights['W_V'],
            W_O=weights['W_O'],
            n_heads=weights['n_heads'],
            W1=weights['W1'],
            b1=weights['b1'],
            W2=weights['W2'],
            b2=weights['b2'],
            gamma1=weights['gamma1'],
            beta1=weights['beta1'],
            gamma2=weights['gamma2'],
            beta2=weights['beta2']
        )
    return x

def get_logits(x, W_out, gamma_final, beta_final):
    x = layer_norm(x, gamma_final, beta_final)
    logits = x @ W_out
    return logits

def forward_model(x, blocks_weights, W_out, gamma_final, beta_final):
    x = transformer_stack(x, blocks_weights)
    logits = get_logits(x, W_out, gamma_final, beta_final)
    return logits