"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    # "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    # "gemini-2.5-pro": {"input": 1.25, "output": 5.00}, # will be deprecated for new users
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
# GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL = "gemini-3.8-flash" # change to gemini-3.8-flash because gemini-2.5 models are deprecated for new users
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    # TODO: Import OpenAI, instantiate client, call chat.completions.create with parameters,
    #       measure execution start/end time, extract text and token usage, and return them.
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    end = time.time()
    latency = end - start
    response_text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }

    text = response.choices[0].message.content or ""
    return text, latency, usage

    raise NotImplementedError("Implement call_openai")


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 3.8 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 3.8 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-3.8-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    # TODO: Initialize Gemini client, set config parameters, call generate_content,
    #       measure latency, extract response text and usage metadata, and return the tuple.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    start = time.time()
    gen_config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=gen_config
    )
    end = time.time()
    latency = end - start
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count
    }
    text = response.text or ""
    return text, latency, usage

    raise NotImplementedError("Implement call_gemini")


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    # TODO: Initialize Anthropic client, create message, measure latency,
    #       extract content text and usage statistics, and return the tuple.
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start = time.time()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    end = time.time()
    latency = end - start
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }
    text = response.content[0].text if response.content else ""
    return text, latency, usage

    raise NotImplementedError("Implement call_anthropic")


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 3.8 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 3.8 Flash (gemini-3.8-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # TODO: Call call_openai with default gpt-4o model
    call_openai_4o = call_openai(prompt, model=OPENAI_MODEL)
    # TODO: Call call_openai with gpt-4o-mini model
    call_openai_4o_mini = call_openai(prompt, model=OPENAI_MINI_MODEL)
    # TODO: Call call_gemini with default gemini-3.8-flash model
    call_gemini_flash = call_gemini(prompt, model=GEMINI_MODEL)
    # TODO: Calculate costs exactly based on input and output token counts using PRICING_1M_TOKENS
    #       Formula: Cost = (input_tokens * input_rate_per_1M + output_tokens * output_rate_per_1M) / 1,000,000
    cost_gpt4o = (call_openai_4o[2]['input_tokens'] * PRICING_1M_TOKENS[OPENAI_MODEL]['input'] + call_openai_4o[2]['output_tokens'] * PRICING_1M_TOKENS[OPENAI_MODEL]['output']) / 1_000_000
    cost_gpt4o_mini = (call_openai_4o_mini[2]['input_tokens'] * PRICING_1M_TOKENS[OPENAI_MINI_MODEL]['input'] + call_openai_4o_mini[2]['output_tokens'] * PRICING_1M_TOKENS[OPENAI_MINI_MODEL]['output']) / 1_000_000
    cost_gemini_flash = (call_gemini_flash[2]['input_tokens'] * PRICING_1M_TOKENS[GEMINI_MODEL]['input'] + call_gemini_flash[2]['output_tokens'] * PRICING_1M_TOKENS[GEMINI_MODEL]['output']) / 1_000_000
    # TODO: Assemble and return the comparison dictionary.
    return {
        "gpt4o": {
            "response": call_openai_4o[0],
            "latency": call_openai_4o[1],
            "cost": cost_gpt4o,
            "input_tokens": call_openai_4o[2]['input_tokens'],
            "output_tokens": call_openai_4o[2]['output_tokens']
        },
        "gpt4o_mini": {
            "response": call_openai_4o_mini[0],
            "latency": call_openai_4o_mini[1],
            "cost": cost_gpt4o_mini,
            "input_tokens": call_openai_4o_mini[2]['input_tokens'],
            "output_tokens": call_openai_4o_mini[2]['output_tokens']
        },
        "gemini_flash": {
            "response": call_gemini_flash[0],
            "latency": call_gemini_flash[1],
            "cost": cost_gemini_flash,
            "input_tokens": call_gemini_flash[2]['input_tokens'],
            "output_tokens": call_gemini_flash[2]['output_tokens']
        }
    }
    raise NotImplementedError("Implement compare_models")


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 3.8 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 3.8.

    Behaviour:
        - Streams response tokens from Gemini 3.8 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        # Sent as the last 3 completed turns plus the new message.
        contents = history + [{"role": "user", "parts": [{"text": user_input}]}]
        print("Gemini: ", end="", flush=True)
        reply = ""
        for chunk in client.models.generate_content_stream(model=GEMINI_MODEL, contents=contents):
            text = chunk.text or ""  # chunk.text is None for chunks without text
            print(text, end="", flush=True)
            reply += text
        print()

        contents.append({"role": "model", "parts": [{"text": reply}]})
        history = contents[-6:]  # 3 turns = 6 messages


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception:
            time.sleep(base_delay * 2 ** attempt)
    return fn()  # final attempt: its exception propagates to the caller


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    return [{**compare_models(prompt), "prompt": prompt} for prompt in prompts]


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    labels = {"gpt4o": "GPT-4o", "gpt4o_mini": "GPT-4o-Mini", "gemini_flash": "Gemini-Flash"}

    def cell(text: str) -> str:
        # Newlines and pipes would break the table row.
        return " ".join(text.split()).replace("|", "\\|")

    rows = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        for key, label in labels.items():
            stats = result[key]
            response = stats["response"]
            short = response[:50] + "..." if len(response) > 50 else response
            rows.append(
                f"| {cell(result['prompt'])} | {label} | {cell(short)} | {stats['latency']:.2f}s "
                f"| {stats['input_tokens']}/{stats['output_tokens']} | ${stats['cost']:.6f} |"
            )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "What's the weather like in Hanoi, Gia Lam today? Is it nice enough to go out?"
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 3.8 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
