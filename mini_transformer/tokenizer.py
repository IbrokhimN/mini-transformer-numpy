# чтение кал в питоне, почти всегда делайте через os
# with open("../data/text.txt", "r") as f:
#     text = f.read().replace("\n", " ")

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "../data/text.txt")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read().replace("\n", " ")

# finding unique characters in the text
unique_set = set(text)
unique_list = sorted(list(unique_set))

# print(unique_list)


# enumerating unique characters to create a mapping from character to index and index to character
ch_to_idx = {ch: i for i, ch in enumerate(unique_list)}
idx_to_ch = {i: ch for i, ch in enumerate(unique_list)}

def encode(text):
    return [ch_to_idx[ch] for ch in text]

def decode(indices):
    return ''.join([idx_to_ch[i] for i in indices])


print(len(unique_list))
'''
encode("Hello") - turns the string "Hello" into a list of indices based on the character to index mapping.
decode([0, 1, 2, 3, 4]) - same here but revesed
'''