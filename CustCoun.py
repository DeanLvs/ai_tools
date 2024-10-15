from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import sys
import os
import hashlib
import threading

sys.path.append('/mnt/sessd/usr/local/lib/python3.10/site-packages/')
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.loaders import (
    TextualInversionLoaderMixin,
)
cache = {}
cache_lock = threading.Lock()
def generate_cache_key(prompt, prompt_2, negative_prompt, negative_prompt_2, lora_scale):
    """
    生成一个唯一的缓存键，基于输入的 prompt、negative prompt 和 lora_scale.
    """
    key_string = f"{prompt}_{prompt_2}_{negative_prompt}_{negative_prompt_2}_{lora_scale}"
    key = hashlib.md5(key_string.encode()).hexdigest()
    return key

def encode_prompt_with_cache(
        sdxlCP,
        prompt,
        prompt_2,
        negative_prompt,
        negative_prompt_2,
        tokenizer=None, tokenizer_2=None, text_encoder=None, text_encoder_2=None,
        do_classifier_free_guidance=True,
        lora_scale=None,
        clip_skip=None,
):
    # 生成缓存键，包含 lora_scale
    cache_key = generate_cache_key(prompt, prompt_2, negative_prompt, negative_prompt_2, lora_scale)

    # 检查缓存是否已经存在
    with cache_lock:
        if cache_key in cache:
            print("使用缓存结果")
            return cache[cache_key]

    # 如果缓存不存在，执行原始逻辑
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt(
        sdxlCP,
        prompt,
        prompt_2,
        negative_prompt,
        negative_prompt_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        do_classifier_free_guidance=do_classifier_free_guidance,
        lora_scale=lora_scale,
        clip_skip=clip_skip
    )

    # 将结果存储到缓存中
    with cache_lock:
        cache[cache_key] = (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)

    return cache[cache_key]

def encode_prompt(
        sdxlCP,
        prompt,
        prompt_2 ,
        negative_prompt,
        negative_prompt_2 ,
        tokenizer= None, tokenizer_2= None, text_encoder=None, text_encoder_2=None,
        do_classifier_free_guidance = True,
        lora_scale = 0.6,
        clip_skip = None,
):
    device = "cpu"

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    if lora_scale is not None:
        sdxlCP._lora_scale = lora_scale

        # dynamically adjust the LoRA scale
        if text_encoder is not None:
            adjust_lora_scale_text_encoder(text_encoder, lora_scale)

        if text_encoder_2 is not None:
            # if not USE_PEFT_BACKEND:
            adjust_lora_scale_text_encoder(text_encoder_2, lora_scale)

    prompt = [prompt] if isinstance(prompt, str) else prompt

    batch_size = len(prompt)

    # Define tokenizers and text encoders
    tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [tokenizer_2]
    text_encoders = ([text_encoder, text_encoder_2] if text_encoder is not None else [text_encoder_2])

    prompt_2 = prompt_2 or prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

    # textual inversion: process multi-vector tokens if necessary
    prompt_embeds_list = []
    prompts = [prompt, prompt_2]
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        if isinstance(sdxlCP, TextualInversionLoaderMixin):
            prompt = sdxlCP.maybe_convert_prompt(prompt, tokenizer)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    # get unconditional embeddings for classifier free guidance
    zero_out_negative_prompt = negative_prompt is None and sdxlCP.config.force_zeros_for_empty_prompt
    if do_classifier_free_guidance and zero_out_negative_prompt:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    elif do_classifier_free_guidance:
        negative_prompt = negative_prompt or ""
        negative_prompt_2 = negative_prompt_2 or negative_prompt

        # normalize str to list
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        negative_prompt_2 = (
            batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
        )

        uncond_tokens: List[str]
        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = [negative_prompt, negative_prompt_2]

        negative_prompt_embeds_list = []
        for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
            if isinstance(sdxlCP, TextualInversionLoaderMixin):
                negative_prompt = sdxlCP.maybe_convert_prompt(negative_prompt, tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
