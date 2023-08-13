import torch
import datasets
from functools import partial


def pad_sequences_and_create_masks(sequences, max_length, padding_value):
    padded_sequences = [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]
    attention_masks = [[1 if token != padding_value else 0 for token in sequence] for sequence in padded_sequences]
    return padded_sequences, attention_masks


def preprocess_dataset(batch, tokenizer, max_length):
    input_ids = []
    label_ids = []
    attention_masks = []
    
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = tokenizer.sep_token_id

    for i in range(len(batch["instruction"])):
        instruction_ids = tokenizer(batch["instruction"][i])["input_ids"][1:] # [1:] is for Llama2
        source_unit_ids = tokenizer.convert_tokens_to_ids([f"v_tok_{u}" for u in batch["src_encodec_0"][i]])
        target_unit_ids = tokenizer.convert_tokens_to_ids([f"v_tok_{u}" for u in batch["tgt_encodec_0"][i]])
        
        context_ids    = instruction_ids + [sep_token_id] + source_unit_ids
        curr_input_ids = context_ids + [sep_token_id] + target_unit_ids
        curr_label_ids = [-100] * len(context_ids) + target_unit_ids + [eos_token_id]

        if len(curr_input_ids) > max_length:
            continue
        
        input_ids.append(curr_input_ids)
        label_ids.append(curr_label_ids)
    
    input_ids, attention_masks = pad_sequences_and_create_masks(input_ids, max_length, -100)
    label_ids, _ = pad_sequences_and_create_masks(label_ids, max_length, -100)
    
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(label_ids),
        "attention_mask": attention_masks
    }


def get_preprocessed_text_vc(dataset_config, tokenizer, split, max_length=2000):
    if split == "test":
        split = "validation"
    dataset = datasets.load_dataset(dataset_config.hf_name, split=split)
    dataset = dataset.filter(lambda s : len(s[f"src_encodec_0"]) <= 800)

    dataset = dataset.map(
        partial(preprocess_dataset, tokenizer=tokenizer, max_length=max_length),
        batched=True
    )
    columns = ["input_ids", "labels", "attention_mask"]
    dataset.set_format(type="torch", columns=columns)
    
    return dataset