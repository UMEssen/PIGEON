"""
Stage 5: Run inference using fine-tuned model via vLLM API.

Uses async concurrent requests to a vLLM OpenAI-compatible endpoint
for efficient batch inference on test data or real medical texts.

The vLLM server should be started separately before running this script.
Example vLLM launch command:

    vllm serve /path/to/finetuned-model \\
        --port 8002 \\
        --max-model-len 8192 \\
        --quantization awq

This script supports two input modes:
    1. Dataset mode: reads a JSON dataset (from Stage 3) and extracts prompts.
    2. Real text mode: reads .txt files from a directory and wraps them in the
       extraction prompt template.

Usage:
    # Inference on test dataset
    python run_inference.py --input /path/to/test.json --output results.csv

    # Inference on real medical texts
    python run_inference.py --input /path/to/texts/ --output results.csv

    # Custom endpoint and concurrency
    python run_inference.py --input test.json --endpoint http://localhost:8002/v1 \\
        --model my-model --max-concurrent 20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *

import argparse
import ast
import asyncio
import json
import os

import aiohttp
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm.asyncio import tqdm as async_tqdm


# =====================================================================
# Prompt template (must match the one used during training in Stage 3)
# =====================================================================

PROMPT_TEMPLATE = """Extrahiere medizinische Informationen aus dem folgenden medizinischen Text und strukturiere sie als JSON.

REGELN:
- Antwort NUR als gültiges JSON-Objekt
- Nur explizit im Text erwähnte Informationen extrahieren
- Fehlende Informationen als leere Werte ("", [], {{}}) angeben
- Vollständige JSON-Struktur gemäß Schema verwenden

JSON-SCHEMA:
{{
  "introduction": {{
    "family_name": "",
    "given_name": "",
    "birth_date": "",
    "gender": "",
    "address_street": "",
    "address_city": "",
    "address_postal_code": "",
    "stationary_type": "",
    "encounter_start_date": "",
    "encounter_end_date": ""
  }},
  "diagnoses": [
    {{
      "type": "main_diagnosis",
      "name": "",
      "icd10gm_code": "",
      "date": ""
    }},
    {{
      "type": "side_diagnosis",
      "name": "",
      "icd10gm_code": "",
      "date": ""
    }}
  ],
  "tumor_informations": [
    {{
      "type": "pathological",
      "stage": "",
      "t": "",
      "n": "",
      "m": "",
      "date": ""
    }},
    {{
      "type": "clinical",
      "t": "",
      "n": "",
      "m": "",
      "date": ""
    }},
    {{
      "type": "histology",
      "histology": "",
      "date": ""
    }},
    {{
      "type": "overall_status",
      "status_de": "",
      "date": ""
    }},
    {{
      "type": "progression",
      "description_de": "",
      "date": ""
    }},
    {{
      "type": "tumor_marker",
      "marker": "",
      "value": 0.0,
      "unit": "",
      "date": ""
    }},
    {{
      "type": "smoking_status",
      "status": "",
      "date": ""
    }},
    {{
      "type": "ecog_performance",
      "score": 0,
      "date": ""
    }},
    {{
      "type": "comorbidities",
      "conditions": [],
      "date": ""
    }},
    {{
      "type": "operations",
      "procedures": [],
      "date": ""
    }},
    {{
      "type": "radiotherapy",
      "procedures": [],
      "date": ""
    }}
  ],
  "medication": [],
  "lab_values": [
    {{
      "lab_name": "",
      "lab_value": 0.0
    }}
  ],
  "free_text": {{
    "lab_values": [
      {{
        "name": "",
        "value": ""
      }}
    ],
    "medications": [],
    "body_values": [],
    "procedures": [
      {{
        "procedure_name": "",
        "ops_code": "",
        "code_type": ""
      }}
    ],
    "diagnoses": [
      {{
        "type": "side_diagnosis",
        "official_name": "",
        "icd10gm_code": ""
      }}
    ]
  }}
}}

MEDIZINISCHER TEXT:
{medical_text}"""


# =====================================================================
# Response parsing
# =====================================================================

def process_generation_response(text: str) -> str:
    """Extract and parse JSON from raw model output.

    The model may wrap its JSON output in markdown fences or include
    extra text. This function finds the outermost JSON object and
    attempts to parse it.

    Args:
        text: Raw text from the model's response.

    Returns:
        A JSON string if parsing succeeded, or the raw string on failure.
    """
    if not isinstance(text, str):
        return json.dumps({"error": "Invalid response type", "response": str(text)})

    # Find the first '{' and last '}' to extract the JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_str = text[start : end + 1]
    else:
        # No JSON object found -- return raw text
        return text.strip()

    # Try standard JSON parsing first
    try:
        parsed = json.loads(json_str)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # Fall back to Python literal eval (handles single quotes, etc.)
    try:
        parsed = ast.literal_eval(json_str)
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=False)
        return str(parsed)
    except (ValueError, SyntaxError):
        pass

    # All parsing failed -- return the extracted substring
    return json_str


# =====================================================================
# Async request handling
# =====================================================================

async def send_request(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    model: str,
) -> str:
    """Send a single inference request to the vLLM OpenAI-compatible API.

    The prompt is sent as a user message in chat completion format.
    A semaphore limits the number of concurrent in-flight requests.

    Args:
        session: aiohttp client session (connection pooling).
        prompt: The user prompt (already formatted with the extraction template).
        semaphore: Asyncio semaphore for concurrency limiting.
        endpoint: Base URL of the vLLM API (e.g. http://localhost:8002/v1).
        model: Model name or path as served by vLLM.

    Returns:
        The model's response text, or an error string.
    """
    url = f"{endpoint}/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0.3,
        "top_p": 0.95,
        "repeat_penalty": 1.2,
    }

    async with semaphore:
        try:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return f"ERROR: HTTP {response.status} - {error_text[:500]}"

                response_json = await response.json()

                if (
                    response_json
                    and "choices" in response_json
                    and len(response_json["choices"]) > 0
                ):
                    return response_json["choices"][0]["message"]["content"]
                else:
                    return "ERROR: No valid response from vLLM API."

        except aiohttp.ClientError as e:
            return f"ERROR: API request failed - {e}"
        except asyncio.TimeoutError:
            return "ERROR: API request timed out."


async def run_batch_inference(
    prompts: list,
    endpoint: str,
    model: str,
    max_concurrent: int = 10,
) -> list:
    """Run inference on a batch of prompts with controlled concurrency.

    Creates async tasks for all prompts and processes them concurrently,
    limited by the semaphore to avoid overwhelming the vLLM server.

    Args:
        prompts: List of formatted prompt strings.
        endpoint: vLLM API base URL.
        model: Model name as served by vLLM.
        max_concurrent: Maximum number of simultaneous requests.

    Returns:
        List of model response strings (same order as prompts).
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, prompt, semaphore, endpoint, model)
            for prompt in prompts
        ]
        # gather with progress bar
        results = await async_tqdm.gather(*tasks, desc="Running inference")

    return results


# =====================================================================
# Prompt preparation
# =====================================================================

def prepare_prompts_from_dataset(dataset_path: str) -> tuple:
    """Load a test dataset (JSON) and extract prompts for inference.

    The dataset is expected to have a 'text' column containing Gemma
    chat-template formatted examples. The prompt is everything up to
    and including '<start_of_turn>model'.

    Args:
        dataset_path: Path to the JSON dataset file.

    Returns:
        Tuple of (prompts, source_files) where source_files is a list of
        'dataset' strings (since examples come from the test set, not files).
    """
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Loaded {len(dataset):,} examples")

    prompts = []
    source_files = []

    for example in dataset:
        text = example["text"]
        # Extract the prompt portion (everything before the model's response).
        # The chat template uses '<start_of_turn>model\n' as the response marker.
        marker = "<start_of_turn>model\n"
        idx = text.find(marker)
        if idx != -1:
            # Include the marker in the prompt so the model knows to generate
            prompt = text[: idx + len(marker)]
        else:
            prompt = text
        # Strip the Gemma chat wrapper to send as a plain user message
        clean = (
            prompt.replace("<bos><start_of_turn>user\n", "")
            .replace("\n<end_of_turn>\n<start_of_turn>model\n", "")
            .strip()
        )
        prompts.append(clean)
        source_files.append("dataset")

    return prompts, source_files


def prepare_prompts_from_real_texts(text_dir: str) -> tuple:
    """Load real medical text files and wrap them in the extraction prompt.

    Reads all .txt files from the given directory and formats each with
    the PROMPT_TEMPLATE used during training.

    Args:
        text_dir: Path to directory containing .txt files.

    Returns:
        Tuple of (prompts, source_files) where source_files contains
        the original filename for each prompt.
    """
    text_path = Path(text_dir)
    if not text_path.exists():
        raise FileNotFoundError(f"Directory not found: {text_dir}")

    txt_files = sorted(text_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {text_dir}")

    print(f"Found {len(txt_files)} text files in: {text_dir}")

    prompts = []
    source_files = []

    for txt_file in txt_files:
        try:
            content = txt_file.read_text(encoding="utf-8").strip()
            if not content:
                print(f"  Skipping empty file: {txt_file.name}")
                continue

            # Format with the extraction prompt template.
            # Note: the template uses {{}} for literal braces, so .format()
            # only substitutes {medical_text}.
            formatted = PROMPT_TEMPLATE.format(medical_text=content)
            prompts.append(formatted)
            source_files.append(txt_file.name)
        except Exception as e:
            print(f"  Error reading {txt_file.name}: {e}")

    print(f"Prepared {len(prompts):,} prompts from real texts")
    return prompts, source_files


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run batch inference via vLLM on test data or real medical texts."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Path to a JSON dataset file (for test-set inference) or a directory "
            "of .txt files (for real-text inference)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "inference_results.csv"),
        help="Path to save the output CSV with results.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=LLM_ENDPOINT_INFERENCE,
        help=f"vLLM API base URL (default: {LLM_ENDPOINT_INFERENCE}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL_INFERENCE,
        help=f"Model name as served by vLLM (default: {LLM_MODEL_INFERENCE}).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT_REQUESTS,
        help=f"Max concurrent API requests (default: {MAX_CONCURRENT_REQUESTS}).",
    )
    args = parser.parse_args()

    ensure_dirs()

    # Determine input mode: directory of texts vs. JSON dataset
    input_path = Path(args.input)

    if input_path.is_dir():
        print("=" * 60)
        print("Mode: Real medical text inference")
        print("=" * 60)
        prompts, source_files = prepare_prompts_from_real_texts(str(input_path))
    elif input_path.is_file() and input_path.suffix == ".json":
        print("=" * 60)
        print("Mode: Test dataset inference")
        print("=" * 60)
        prompts, source_files = prepare_prompts_from_dataset(str(input_path))
    else:
        raise ValueError(
            f"Input must be a .json file or a directory of .txt files. Got: {args.input}"
        )

    if not prompts:
        print("No prompts to process. Exiting.")
        return

    print(f"\nTotal prompts: {len(prompts):,}")
    print(f"Endpoint:      {args.endpoint}")
    print(f"Model:         {args.model}")
    print(f"Concurrency:   {args.max_concurrent}")

    # Run async inference
    print("\n" + "=" * 60)
    print("Running inference...")
    print("=" * 60)
    generations = asyncio.run(
        run_batch_inference(
            prompts,
            endpoint=args.endpoint,
            model=args.model,
            max_concurrent=args.max_concurrent,
        )
    )

    # Parse model outputs into structured JSON
    print("\nParsing model outputs...")
    parsed_generations = [process_generation_response(gen) for gen in generations]

    # Save results to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "prompt": prompts,
            "generation": generations,
            "parsed_generation": parsed_generations,
            "source_file": source_files,
        }
    )
    df.to_csv(output_path, index=False)

    print(f"\nInference complete.")
    print(f"Results saved to: {output_path}")
    print(f"Processed {len(prompts):,} examples using model: {args.model}")

    # Quick summary of errors
    error_count = sum(1 for g in generations if isinstance(g, str) and g.startswith("ERROR"))
    if error_count > 0:
        print(f"Warning: {error_count} requests returned errors.")


if __name__ == "__main__":
    main()
