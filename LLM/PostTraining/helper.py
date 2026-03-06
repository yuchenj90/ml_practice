from transformers import AutoModelForCausalLM, AutoTokenizer
from Qwen3 import Qwen3Tokenizer
from Qwen3 import Qwen3Model, QWEN_CONFIG_06_B, QWEN_CONFIG_4_B_INSTRUCT
from collections import OrderedDict
from pathlib import Path
import os
import torch

def load_model_weights(tokenizer_path, model_path, model_config, model_name=None, save_path=''):
    # try load tokenizer
    try:
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
        print(f"Successfully loaded tokenizer from {tokenizer_path}!")
    except:
        print(f"Tokenizer Path does not exist, loading using model_name {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Save the backend tokenizer to a single JSON file
        print(f"Saving tokenizer to {save_path}...")
        tokenizer_save_path = save_path + model_name + '.json'
        pathdir = Path('/'.join((tokenizer_save_path.split('/')[:-1])))
        if not pathdir.is_dir():
            os.mkdir(pathdir)
        if hasattr(tokenizer, 'backend_tokenizer'):
            tokenizer.backend_tokenizer.save(tokenizer_save_path)
            print(f"Tokenizer saved to {tokenizer_save_path}")
        else:
            print("Tokenizer does not have a backend_tokenizer attribute or is not a fast tokenizer.")
            
    # try load model
    try:
        print(f"Using {model_config} load model...")
        model = Qwen3Model(model_config)
        print(f"Successfully loaded model from config {model_config}.")
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded weights from {model_path}")
    except:
        print(f"Model does not exist, loading using model_name {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print(f"Successfully loaded model from {model_name}")
        model_param_dict = OrderedDict()
        for k,v in model.state_dict().items():
            if k=='model.embed_tokens.weight':
                model_param_dict['tok_emb.weight'] = v
            elif k == 'model.norm.weight':
                model_param_dict['final_norm.scale'] = v
            elif k == 'lm_head.weight':
                model_param_dict['out_head.weight'] = v
            else:
                k = k.replace('model.layers', 'trf_blocks')
                k = k.replace('self_attn', 'att')
                k = k.replace('q_proj', 'W_query')
                k = k.replace('k_proj', 'W_key')
                k = k.replace('v_proj', 'W_value')
                k = k.replace('o_proj', 'out_proj')
                k = k.replace('norm.weight', 'norm.scale')
                k = k.replace('mlp.gate_proj', 'ff.fc1')
                k = k.replace('mlp.up_proj', 'ff.fc2')
                k = k.replace('mlp.down_proj', 'ff.fc3')
                k = k.replace('input_layernorm', 'norm1')
                k = k.replace('post_attention_layernorm', 'norm2')
                model_param_dict[k] = v
                
        # 4. Save the model's state_dict to a .pth file
        print(f"Saving model to {save_path}...")
        output_pth_path = save_path + model_name + ".pth"
        pathdir = Path('/'.join((output_pth_path.split('/')[:-1])))
        if not pathdir.is_dir():
            os.mkdir(pathdir)
        torch.save(model_param_dict, output_pth_path)
        print(f"Model successfully saved to {output_pth_path}")
    return tokenizer, model
