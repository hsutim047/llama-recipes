import copy

from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("../llama2_hf_models/7B")
pretrained = LlamaForCausalLM.from_pretrained("../llama2_hf_models/7B")

print(tokenizer.tokenize("Hello world v_tok_10v_tok_1<sep>"))
add_tokens = [f"v_tok_{u}" for u in range(1024 * 9)]
origin_vocab_size = tokenizer.vocab_size
print("===ADD TOKEN===")
num_added_toks = tokenizer.add_tokens(add_tokens)
num_added_special_toks = tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>"})
print('We have added', num_added_toks + num_added_special_toks, 'tokens')
print(origin_vocab_size, num_added_toks + num_added_special_toks, len(tokenizer))
print(tokenizer.tokenize("Hello world v_tok_10v_tok_1<sep>"))
print("===============")
# reshape pretraining embedding
pretrained.resize_token_embeddings(origin_vocab_size + num_added_toks + num_added_special_toks)
tokenizer.push_to_hub("tsuyuan/Llama-2-7b-unit_random_embed")
pretrained.push_to_hub("tsuyuan/Llama-2-7b-unit_random_embed")
input_embedding = pretrained.get_input_embeddings()
state_dict_weight = input_embedding.state_dict()['weight']
print(state_dict_weight.shape, state_dict_weight[10:100].shape)
state_dict_weight[origin_vocab_size:origin_vocab_size + num_added_toks + num_added_special_toks] = copy.copy(
    state_dict_weight[100:100 + num_added_toks + num_added_special_toks])
pretrained.set_input_embeddings(input_embedding)
print("===============")

tokenizer.push_to_hub("tsuyuan/Llama-2-7b-unit")
pretrained.push_to_hub("tsuyuan/Llama-2-7b-unit")