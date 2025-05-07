from data1.serialize import serialize_arr, SerializerSettings
import openai
import tiktoken
import numpy as np
from jax import grad, vmap
import google.generativeai as genai

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

def tokenize_fn(string, model):
    """
    Tokenize a string using the tokenizer for the specified GPT model.
    """
    encoding = tiktoken.encoding_for_model(model)
    return encoding.encode(string)

def get_allowed_ids(strings, model):
    """
    Tokenize a list of strings using the tokenizer for the specified GPT model.
    """
    encoding = tiktoken.encoding_for_model(model)
    ids = []
    for s in strings:
        ids.extend(encoding.encode(s))
    return ids

def gemini_completion_fn(model, input_str, steps, settings, num_samples, temp, whether_blanket=True, genai_key=None):
    """
    Generate text completions using Google's Gemini API.
    """
    if genai_key is not None:
        genai.configure(api_key=genai_key, transport='rest')

    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)]
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [t for t in allowed_tokens if len(t) > 0]

    if not whether_blanket:
        input_str = input_str.replace(" ", "")

    if model not in ['gemini-1.0-pro', 'gemini-pro']:
        logit_bias = {id: 30 for id in get_allowed_ids(allowed_tokens, model)}

    if model in ['gemini-1.0-pro', 'gemini-pro']:
        gemini_sys_message = (
            f"You are a helpful assistant that performs time series predictions. "
            f"The user will provide a sequence and you will predict the remaining sequence for {steps * 10} steps. "
            f"The sequence is represented by decimal strings separated by commas, and each step consists of contents between two commas."
        )
        extra_input = (
            "Please continue the following sequence without producing any additional text. "
            "Do not include phrases like 'the next terms in the sequence are'. Just return the numbers.\n"
            "Sequence:\n"
        )

        content_fin = []
        model = genai.GenerativeModel(model)
        for i in range(num_samples):
            print("Index:", i)
            response = model.generate_content(
                contents=gemini_sys_message + extra_input + input_str + settings.time_sep,
                generation_config=genai.types.GenerationConfig(
                    temperature=temp,
                ),
                safety_settings=safety_settings
            )
            tmp = response.text
            if not whether_blanket:
                tmp = ' '.join(response.text)
            content_fin.append(tmp)
        return content_fin
    else:
        assert False

def gpt_completion_fn(model, input_str, steps, settings, num_samples, temp, whether_blanket=True):
    """
    Generate text completions using OpenAI GPT models.
    """
    avg_tokens_per_step = len(tokenize_fn(input_str, model)) / len(input_str.split(settings.time_sep))

    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)]
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [t for t in allowed_tokens if len(t) > 0]

    logit_bias = {}
    if model not in ['gpt-3.5-turbo', 'gpt-4']:
        logit_bias = {id: 30 for id in get_allowed_ids(allowed_tokens, model)}

    if model in ['gpt-3.5-turbo', 'gpt-4']:
        chatgpt_sys_message = (
            "You are a helpful assistant that performs time series predictions. "
            "The user will provide a sequence and you will predict the remaining sequence. "
            "The sequence is represented by decimal strings separated by commas."
        )
        extra_input = (
            "Please continue the following sequence without producing any additional text. "
            "Do not include phrases like 'the next terms in the sequence are'. Just return the numbers.\n"
            "Sequence:\n"
        )
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": chatgpt_sys_message},
                {"role": "user", "content": extra_input + input_str + settings.time_sep}
            ],
            max_tokens=int(avg_tokens_per_step * steps),
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples
        )
        return [choice.message.content for choice in response.choices]
    else:
        response = openai.Completion.create(
            model=model,
            prompt=input_str,
            max_tokens=int(avg_tokens_per_step * steps),
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples
        )
        return [choice.text for choice in response.choices]

def gpt_nll_fn(model, input_arr, target_arr, settings: SerializerSettings, transform, count_seps=True, temp=1):
    """
    Compute the Negative Log-Likelihood (NLL) per dimension under the LLM.
    """
    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    assert input_str.endswith(settings.time_sep), f'Input string must end with {settings.time_sep}, got {input_str}'

    full_series = input_str + target_str
    response = openai.Completion.create(
        model=model,
        prompt=full_series,
        logprobs=5,
        max_tokens=0,
        echo=True,
        temperature=temp
    )
    logprobs = np.array(response['choices'][0].logprobs.token_logprobs, dtype=np.float32)
    tokens = np.array(response['choices'][0].logprobs.tokens)
    top5logprobs = response['choices'][0].logprobs.top_logprobs

    seps = tokens == settings.time_sep
    target_start = np.argmax(np.cumsum(seps) == len(input_arr)) + 1
    logprobs = logprobs[target_start:]
    tokens = tokens[target_start:]
    top5logprobs = top5logprobs[target_start:]
    seps = tokens == settings.time_sep

    assert len(logprobs[seps]) == len(target_arr), (
        f'There should be one separator per target. Got {len(logprobs[seps])} separators and {len(target_arr)} targets.'
    )

    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)]
    allowed_tokens += [
        settings.time_sep,
        settings.plus_sign,
        settings.minus_sign,
        settings.bit_sep + settings.decimal_point
    ]
    allowed_tokens = {t for t in allowed_tokens if len(t) > 0}

    p_extra = np.array([
        sum(np.exp(lp) for k, lp in top5logprobs[i].items() if k not in allowed_tokens)
        for i in range(len(top5logprobs))
    ])
    if settings.bit_sep == '':
        p_extra = 0

    adjusted_logprobs = logprobs - np.log(1 - p_extra)
    digits_bits = -adjusted_logprobs[~seps].sum()
    seps_bits = -adjusted_logprobs[seps].sum()

    BPD = digits_bits / len(target_arr)
    if count_seps:
        BPD += seps_bits / len(target_arr)

    transformed_nll = BPD - settings.prec * np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    return transformed_nll - avg_logdet_dydx
