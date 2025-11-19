import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import system_prompt_utils


def get_model(model_path, device='cuda', dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)

    model.requires_grad_(False)
    model.eval()
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    return model, tokenizer


def get_input_template(user_prompt,
                       target_response,
                       len_adv_tokens,
                       tokenizer,
                       model_name,
                       use_llama_system_prompt=False,
                       self_propagating=False,
                       use_system_prompt=True):

    model_name = model_name.lower()
    flag = 'r2d2' in model_name

    if 'zephyr' in model_name:
        model_name = 'zephyr'
        system_prompt = system_prompt_utils.ZEPHYR
    elif 'vicuna' in model_name:
        model_name = 'vicuna'
        system_prompt = system_prompt_utils.VICUNA
    elif 'llama-3' in model_name:
        model_name = 'llama3'
        system_prompt = system_prompt_utils.LLAMA3
    elif 'llama' in model_name:
        model_name = 'llama-2-chat'
        system_prompt = system_prompt_utils.LLAMA2
    elif 'mistral' in model_name:
        model_name = 'mistral-instruct'
        system_prompt = system_prompt_utils.MISTRAL
    else:
        raise ValueError('model not supported yet')

    if use_llama_system_prompt:
        system_prompt = system_prompt_utils.LLAMA2

    adv_tokens = ' !' * len_adv_tokens

    # For self-propagating attacks, the target includes the adversarial suffix
    # so it appears in the output and can infect subsequent agents
    if self_propagating:
        assistant_content = adv_tokens + target_response
    else:
        assistant_content = target_response

    # Build messages with optional system prompt
    messages = []
    if use_system_prompt:
        messages.append({
            'role': 'system',
            'content': system_prompt
        })
    messages.extend([{
        'role': 'user',
        'content': user_prompt + adv_tokens
    }, {
        'role': 'assistant',
        'content': assistant_content
    }])

    if model_name != 'llama3':
        tokenizer.chat_template = get_chat_template(model_name)
    if flag:
        messages = messages[1:]
    string = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    string = assistant_content.join(string.split(assistant_content)[:-1])
    string = string + assistant_content
    # flag = not string.startswith('<s>')
    input_ids = tokenizer(string, add_special_tokens=flag).input_ids

    target_stop = len(input_ids)
    target_start = None
    adv_start = None
    adv_stop = None

    for i in range(target_stop, 0, -1):
        # Look for target start (assistant content)
        decoded = tokenizer.decode(input_ids[i:])
        # Handle potential whitespace differences in tokenization
        # Try exact match first, then stripped versions
        if target_start is None:
            if (decoded == assistant_content or
                decoded.lstrip() == assistant_content.lstrip() or
                decoded == assistant_content.lstrip()):
                target_start = i
        # Look for adversarial tokens (in user input, not assistant)
        # In self-propagating mode, adv_tokens appears in both user and assistant
        # We want the one in the user message, so skip if we're in the assistant part
        if adv_start is None:
            # Try multiple variants to handle whitespace differences in tokenization
            # Check for: "! ! ! !" or " ! ! ! !" (with or without leading space)
            adv_stripped = adv_tokens.strip()
            adv_no_leading_space = adv_tokens[1:] if adv_tokens.startswith(' ') else adv_tokens

            if (adv_stripped in decoded or
                adv_no_leading_space in decoded or
                adv_tokens in decoded):
                # Make sure we're not finding the adv_tokens in the assistant message
                if target_start is None or i < target_start:
                    adv_start, adv_stop = i, i + len_adv_tokens
        # Break only when both are found
        if target_start is not None and adv_start is not None:
            break

    # Check if we found both required positions
    if target_start is None:
        raise ValueError(f"Could not find target response in tokenized input. "
                        f"assistant_content={repr(assistant_content)}, "
                        f"decoded end={repr(tokenizer.decode(input_ids[-50:]))}")
    if adv_start is None:
        raise ValueError(f"Could not find adversarial tokens in tokenized input. "
                        f"adv_tokens={repr(adv_tokens)}")

    slices = {
        'adv_slice': slice(adv_start, adv_stop),
        'target_slice': slice(target_start, target_stop),
        'loss_slice': slice(target_start - 1, target_stop - 1)
    }

    # Note: We don't assert exact string matches because tokenize→decode
    # is not always reversible due to BPE merging, whitespace handling, etc.
    # The slice positions are what matter for the attack.
    input_ids = torch.tensor(input_ids)
    return string, input_ids, slices


def get_chat_template(model_name):
    chat_temp_path = f'chat_templates/{model_name}.jinja'
    chat_template = open(chat_temp_path).read()
    chat_template = chat_template.replace('    ', '')
    chat_template = chat_template.replace('}\n', '}')
    return chat_template
