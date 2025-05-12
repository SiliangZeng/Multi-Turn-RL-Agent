from importlib.util import find_spec
from typing import Dict, Any, Union, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_liger_available() -> bool:
    return find_spec("liger_kernel") is not None

def get_model(
    model_name: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    subfolder: Optional[str] = None
) -> Any:
    if model_kwargs is None:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
    if subfolder:
        model_kwargs["subfolder"] = subfolder

    if is_liger_available():
        print("Using Liger kernel")
        from liger_kernel.transformers import AutoLigerKernelForCausalLM  # type: ignore
        return AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

def get_tokenizer(model_name: str, subfolder: Union[str, None] = None) -> Any:
    tokenizer = None
    kwargs = {}
    if subfolder is not None:
        kwargs["subfolder"] = subfolder

    if "Instruct" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name + "-Instruct", **kwargs)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

    if not hasattr(tokenizer, "pad_token"):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not hasattr(tokenizer, "chat_template"):
        raise ValueError(f"Tokenizer for model {model_name} does not have chat_template attribute, \
                            and could not find a tokenizer with the same name as the model with suffix \
                            '-Instruct'. Please provide a tokenizer with the chat_template attribute.")
    return tokenizer


def get_model_and_tokenizer(
    model_name: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    subfolder: Optional[str] = None
) -> Tuple[Any, Any]:
    model = get_model(model_name, model_kwargs, subfolder)
    tokenizer = get_tokenizer(model_name, subfolder)
    return model, tokenizer
