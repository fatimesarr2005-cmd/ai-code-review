from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("test_model")


DEFAULT_OLD_FILE = "import torch"
DEFAULT_DIFF_HUNK = "@@ -1 +1,2 @@\n import torch\n +import torch.nn as nn"


def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA demandé mais indisponible. Basculage vers CPU.")
        requested = "cpu"
    return torch.device(requested)

@staticmethod
def build_prompt(old_file: str, diff_hunk: str) -> str:
    old_file = old_file.strip()
    diff_hunk = diff_hunk.strip()

    return (
        "You are an AI code reviewer.\n"
        "Read the old file and the diff hunk, then generate a short and useful code review comment.\n\n"
        "old_file:\n"
        f"{old_file}\n\n"
        "diff_hunk:\n"
        f"{diff_hunk}\n\n"
        "review_comment:\n"
    )


def load_bundle(
    model_name_or_path: str,
    device: torch.device,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
):
    load_kwargs: Dict[str, object] = {"local_files_only": local_files_only}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    LOGGER.info("Chargement du tokenizer depuis %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)

    LOGGER.info("Chargement du modèle depuis %s", model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **load_kwargs)

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(
        tokenizer, "eos_token", None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model.to(device)
    except RuntimeError as exc:
        if device.type == "cuda":
            LOGGER.warning("Échec du placement sur GPU (%s). Repli sur CPU.", exc)
            device = torch.device("cpu")
            model.to(device)
        else:
            raise

    model.eval()
    return tokenizer, model, device


def generate_comment(
    tokenizer,
    model,
    device: torch.device,
    old_file: str,
    diff_hunk: str,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    prompt = build_prompt(old_file, diff_hunk)
    LOGGER.info("Préparation du prompt (%d caractères).", len(prompt))

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "early_stopping": True,
        "no_repeat_ngram_size": 2,
    }

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teste localement un modèle de revue de code seq2seq."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_PATH") or os.getenv("MODEL_NAME", "microsoft/codereviewer"),
        help="Nom Hugging Face ou chemin local du modèle.",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("MODEL_DEVICE", "auto"),
        choices=["auto", "cpu", "cuda"],
        help="Device d'inférence.",
    )
    parser.add_argument(
        "--old-file",
        default=DEFAULT_OLD_FILE,
        help="Ancienne version du fichier.",
    )
    parser.add_argument(
        "--diff-hunk",
        default=DEFAULT_DIFF_HUNK,
        help="Diff hunk à analyser.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=int(os.getenv("MAX_INPUT_LENGTH", "512")),
        help="Longueur maximale d'entrée après tokenisation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("MAX_NEW_TOKENS", "96")),
        help="Nombre maximal de nouveaux tokens générés.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=int(os.getenv("NUM_BEAMS", "4")),
        help="Largeur de beam search.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="N'utiliser que les fichiers locaux déjà présents.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("MODEL_CACHE_DIR"),
        help="Répertoire de cache optionnel pour Hugging Face.",
    )
    return parser.parse_args()
def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    LOGGER.info("===== Configuration =====")
    LOGGER.info("Modèle        : %s", args.model)
    LOGGER.info("Device        : %s", device)
    LOGGER.info("Local only    : %s", args.local_files_only)
    LOGGER.info("Max input len : %s", args.max_input_length)
    LOGGER.info("Max new toks  : %s", args.max_new_tokens)
    LOGGER.info("Num beams     : %s", args.num_beams)

    try:
        tokenizer, model, device = load_bundle(
            model_name_or_path=args.model,
            device=device,
            local_files_only=args.local_files_only,
            cache_dir=args.cache_dir,
        )
        output = generate_comment(
            tokenizer=tokenizer,
            model=model,
            device=device,
            old_file=args.old_file,
            diff_hunk=args.diff_hunk,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
    except Exception as exc:
        LOGGER.exception("Échec du test modèle.")
        print(f"\nERREUR: {exc}", file=sys.stderr)
        return 1

    print("\n===== Exemple d'entrée =====")
    print("old_file:")
    print(args.old_file)
    print("\ndiff_hunk:")
    print(args.diff_hunk)

    print("\n===== Sortie générée =====")
    print(output if output else "[sortie vide]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 echo "# test"