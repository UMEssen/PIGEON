"""
Dataset creation entry point -- Stage 2 of the PIGEON pipeline.

This script orchestrates batch generation of synthetic medical texts
using the ``TextGenerator`` class.  It supports four generation modes:

1. **arztbrief**       -- Encounter-level discharge letters.
2. **arztbrief_pat**   -- Patient-level discharge letters.
3. **free_text**       -- Free-form clinical notes (encounter-based).
4. **vd_free_text**    -- Verlaufsdokumentation (progress notes).

Each mode reads a JSON recipe file that specifies which sections to
include and how many data items to allocate per section.  Results are
streamed to CSV files row-by-row so that partial runs are recoverable.

Usage::

    python -m src.stage2_dataset_generation.create_dataset \\
        --arztbrief --arztbrief-pat \\
        --max-recipes 1000 --batch-size 16

WHY async batching?
    Each generation may invoke 1-2 LLM calls (Anamnese + Epikrise).
    Running a batch of 16-32 generations concurrently keeps the vLLM
    server saturated without overwhelming it.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import pickle
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import central config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Also add this directory so sibling modules can be found
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

from text_generator import TextGenerator


# =========================================================================
# Recipe loading
# =========================================================================

def _load_recipes(json_path: str) -> List[Dict]:
    """Load recipe JSON with optional pickle caching.

    WHY pickle cache?
        The recipe JSON can be several hundred MB for large cohorts.
        A pickle cache avoids re-parsing on every restart.
    """
    pkl_path = json_path + ".pkl"
    if (
        os.path.exists(pkl_path)
        and os.path.getmtime(pkl_path) >= os.path.getmtime(json_path)
    ):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    with open(json_path, "r") as f:
        recipes = json.load(f)
    with open(pkl_path, "wb") as f:
        pickle.dump(recipes, f)
    return recipes


# =========================================================================
# Per-item task wrappers
# =========================================================================

async def _generate_arztbrief_task(
    recipe: Dict,
    df_real_ab: pd.DataFrame,
    generator: TextGenerator,
) -> Dict[str, Any]:
    """Generate one encounter-level Arztbrief from a recipe."""
    encounter_id = recipe["encounter_id"]
    try:
        real_ab = ""
        rows = df_real_ab[df_real_ab["encounter_id"] == encounter_id]
        if not rows.empty:
            path = rows["file_path"].values[0]
            with open(path, "r") as f:
                real_ab = f.read()

        result = await generator.generate_arztbrief(
            encounter_id=encounter_id,
            sections=recipe.get("sections_to_include", []),
            jargon_variant=random.randint(0, 2),
            recipe_dict=recipe.get("recipe_details", {}),
            real_ab=real_ab,
        )
        return {
            "encounter_id": encounter_id,
            "gen_text": result.get("text", ""),
            "combined_labels": result.get("labels", ""),
            "full_text": result.get("synthetic_text", ""),
            "real_text": result.get("real_text", ""),
            "error": "",
            "prompt": result.get("prompt", ""),
        }
    except Exception as e:
        return {
            "encounter_id": encounter_id,
            "gen_text": "",
            "combined_labels": "",
            "full_text": "",
            "real_text": "",
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "prompt": "",
        }


async def _generate_pat_arztbrief_task(
    recipe: Dict,
    df_real_ab: pd.DataFrame,
    generator: TextGenerator,
) -> Dict[str, Any]:
    """Generate one patient-level Arztbrief from a recipe."""
    patient_id = recipe["patient_id"]
    try:
        real_ab = ""
        rows = df_real_ab[df_real_ab["patient_id"] == patient_id]
        if not rows.empty:
            path = rows["file_path"].values[0]
            with open(path, "r") as f:
                real_ab = f.read()

        gen = TextGenerator(
            base="patient",
            llm_endpoint_primary=config.LLM_ENDPOINT_PRIMARY,
            llm_endpoint_secondary=config.LLM_ENDPOINT_SECONDARY,
        )
        result = await gen.generate_arztbrief_patient(
            patient_id=str(patient_id),
            sections=recipe.get("sections_to_include", []),
            jargon_variant=random.randint(0, 2),
            recipe_dict=recipe.get("recipe_details", {}),
            real_ab=real_ab,
        )
        return {
            "patient_id": patient_id,
            "gen_text": result.get("text", ""),
            "combined_labels": result.get("labels", ""),
            "full_text": result.get("synthetic_text", ""),
            "real_text": result.get("real_text", ""),
            "error": "",
            "prompt": result.get("prompt", ""),
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "gen_text": "",
            "combined_labels": "",
            "full_text": "",
            "real_text": "",
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "prompt": "",
        }


async def _generate_free_text_task(
    recipe: Dict,
    generator: TextGenerator,
) -> Dict[str, Any]:
    """Generate one free-text clinical note."""
    encounter_id = recipe["encounter_id"]
    try:
        raw = generator.loader.load_data(encounter_id)
        result = await generator.generate_free_text(
            input_data=raw,
            jargon_variant=random.randint(0, 2),
        )
        return {
            "encounter_id": encounter_id,
            "gen_text": result.get("text", ""),
            "combined_labels": result.get("labels", ""),
            "error": "",
        }
    except Exception as e:
        return {
            "encounter_id": encounter_id,
            "gen_text": "",
            "combined_labels": "",
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


# =========================================================================
# Main async loop
# =========================================================================

async def main_async(
    do_arztbrief: bool = True,
    do_arztbrief_pat: bool = True,
    do_free_text: bool = True,
    do_vd_free_text: bool = True,
    max_recipes: int = 15000,
    batch_size: int = 32,
    recipe_path_enc: str = "",
    recipe_path_pat: str = "",
    real_ab_enc_path: str = "",
    real_ab_pat_path: str = "",
    output_dir: str = "",
) -> None:
    """Run batch generation across all enabled modes."""
    start = time.time()

    out = Path(output_dir) if output_dir else config.GENERATED_TEXTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    # -- Encounter generator ----------------------------------------------
    gen_enc = TextGenerator(
        base="encounter",
        llm_endpoint_primary=config.LLM_ENDPOINT_PRIMARY,
        llm_endpoint_secondary=config.LLM_ENDPOINT_SECONDARY,
    )

    # -- Load recipes & real texts ----------------------------------------
    recipes_enc: List[Dict] = []
    recipes_pat: List[Dict] = []
    df_real_enc = pd.DataFrame()
    df_real_pat = pd.DataFrame()

    if (do_arztbrief or do_free_text or do_vd_free_text) and recipe_path_enc:
        recipes_enc = _load_recipes(recipe_path_enc)
    if do_arztbrief_pat and recipe_path_pat:
        recipes_pat = _load_recipes(recipe_path_pat)
    if do_arztbrief and real_ab_enc_path:
        df_real_enc = pd.read_csv(real_ab_enc_path)
    if do_arztbrief_pat and real_ab_pat_path:
        df_real_pat = pd.read_csv(real_ab_pat_path)

    # -- Shuffle and limit ------------------------------------------------
    random.shuffle(recipes_enc)
    recipes_enc = recipes_enc[:max_recipes]
    recipes_pat = recipes_pat[:max_recipes]

    # -- CSV writers ------------------------------------------------------
    writers: Dict[str, Any] = {}
    files: Dict[str, Any] = {}

    def _open_csv(name: str, fieldnames: List[str]):
        path = out / f"{name}.csv"
        fh = open(path, "w", newline="", encoding="utf-8")
        w = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, escapechar="\\")
        w.writeheader()
        writers[name] = w
        files[name] = fh

    ab_fields = ["encounter_id", "gen_text", "combined_labels", "full_text", "real_text", "error", "prompt"]
    ab_pat_fields = ["patient_id", "gen_text", "combined_labels", "full_text", "real_text", "error", "prompt"]
    ft_fields = ["encounter_id", "gen_text", "combined_labels", "error"]

    if do_arztbrief:
        _open_csv("arztbriefs", ab_fields)
    if do_arztbrief_pat:
        _open_csv("arztbriefs_pat", ab_pat_fields)
    if do_free_text:
        _open_csv("free_texts", ft_fields)
    if do_vd_free_text:
        _open_csv("vd_free_texts", ft_fields)

    # -- Prepare batches --------------------------------------------------
    def _batch(lst):
        return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]

    ab_batches = _batch(recipes_enc) if do_arztbrief else []
    ab_pat_batches = _batch(recipes_pat) if do_arztbrief_pat else []
    ft_batches = _batch(recipes_enc) if do_free_text else []
    vd_batches = _batch(recipes_enc) if do_vd_free_text else []

    max_batches = max(
        len(ab_batches), len(ab_pat_batches), len(ft_batches), len(vd_batches), 1
    )

    pbars = {
        "ab": tqdm(total=len(ab_batches), desc="Arztbrief", position=0, leave=True) if do_arztbrief else None,
        "ab_pat": tqdm(total=len(ab_pat_batches), desc="Arztbrief Pat", position=1, leave=True) if do_arztbrief_pat else None,
        "ft": tqdm(total=len(ft_batches), desc="Free Text", position=2, leave=True) if do_free_text else None,
        "vd": tqdm(total=len(vd_batches), desc="VD Free Text", position=3, leave=True) if do_vd_free_text else None,
    }

    # -- Batch loop -------------------------------------------------------
    for bi in range(max_batches):
        tasks = []

        if do_arztbrief and bi < len(ab_batches):
            tasks += [
                _generate_arztbrief_task(r, df_real_enc, gen_enc)
                for r in ab_batches[bi]
            ]
        if do_arztbrief_pat and bi < len(ab_pat_batches):
            tasks += [
                _generate_pat_arztbrief_task(r, df_real_pat, gen_enc)
                for r in ab_pat_batches[bi]
            ]
        if do_free_text and bi < len(ft_batches):
            tasks += [
                _generate_free_text_task(r, gen_enc)
                for r in ft_batches[bi]
            ]
        if do_vd_free_text and bi < len(vd_batches):
            tasks += [
                _generate_free_text_task(r, gen_enc)
                for r in vd_batches[bi]
            ]

        results = await asyncio.gather(*tasks)
        idx = 0

        if do_arztbrief and bi < len(ab_batches):
            for _ in ab_batches[bi]:
                writers["arztbriefs"].writerow(results[idx])
                idx += 1
            pbars["ab"].update(1)

        if do_arztbrief_pat and bi < len(ab_pat_batches):
            for _ in ab_pat_batches[bi]:
                writers["arztbriefs_pat"].writerow(results[idx])
                idx += 1
            pbars["ab_pat"].update(1)

        if do_free_text and bi < len(ft_batches):
            for _ in ft_batches[bi]:
                writers["free_texts"].writerow(results[idx])
                idx += 1
            pbars["ft"].update(1)

        if do_vd_free_text and bi < len(vd_batches):
            for _ in vd_batches[bi]:
                writers["vd_free_texts"].writerow(results[idx])
                idx += 1
            pbars["vd"].update(1)

    # -- Cleanup ----------------------------------------------------------
    for pb in pbars.values():
        if pb:
            pb.close()
    for fh in files.values():
        fh.close()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


# =========================================================================
# CLI entry point
# =========================================================================

def main():
    """Parse arguments and run the async generation loop."""
    parser = argparse.ArgumentParser(
        description="PIGEON Stage 2: Synthetic medical text generation.",
    )
    parser.add_argument("--arztbrief", action="store_true", help="Generate encounter-level Arztbriefe")
    parser.add_argument("--arztbrief-pat", action="store_true", help="Generate patient-level Arztbriefe")
    parser.add_argument("--free-text", action="store_true", help="Generate free-form clinical notes")
    parser.add_argument("--vd-free-text", action="store_true", help="Generate Verlaufsdokumentation notes")
    parser.add_argument("--all", action="store_true", help="Enable all generation modes")
    parser.add_argument("--max-recipes", type=int, default=15000, help="Max recipes per mode")
    parser.add_argument("--batch-size", type=int, default=32, help="Async batch size")
    parser.add_argument("--recipe-enc", type=str, default="", help="Path to encounter recipe JSON")
    parser.add_argument("--recipe-pat", type=str, default="", help="Path to patient recipe JSON")
    parser.add_argument("--real-ab-enc", type=str, default="", help="Path to real encounter AB CSV")
    parser.add_argument("--real-ab-pat", type=str, default="", help="Path to real patient AB CSV")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory override")

    args = parser.parse_args()

    do_all = args.all
    asyncio.run(
        main_async(
            do_arztbrief=do_all or args.arztbrief,
            do_arztbrief_pat=do_all or args.arztbrief_pat,
            do_free_text=do_all or args.free_text,
            do_vd_free_text=do_all or args.vd_free_text,
            max_recipes=args.max_recipes,
            batch_size=args.batch_size,
            recipe_path_enc=args.recipe_enc,
            recipe_path_pat=args.recipe_pat,
            real_ab_enc_path=args.real_ab_enc,
            real_ab_pat_path=args.real_ab_pat,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
