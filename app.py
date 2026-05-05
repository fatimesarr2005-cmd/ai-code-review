from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from typing import Dict, Optional

import re

import torch
from flask import Flask, flash, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from werkzeug.exceptions import RequestEntityTooLarge

from github_integration import (
    get_file_content,
    get_pr_files,
    post_pr_comment,
    verify_signature,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # L'application fonctionne aussi sans python-dotenv.
    pass


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("ai_code_review_app")


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning(
            "Valeur entière invalide pour %s=%r. Valeur par défaut=%s.",
            name,
            raw,
            default,
        )
        return default


def resolve_device(requested: Optional[str] = None) -> torch.device:
    chosen = (requested or os.getenv("MODEL_DEVICE", "auto")).strip().lower()
    if chosen == "auto":
        chosen = "cuda" if torch.cuda.is_available() else "cpu"
    if chosen == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA demandé mais indisponible. Basculage sur CPU.")
        chosen = "cpu"
    return torch.device(chosen)


@dataclass
class GenerationResult:
    text: str
    truncated: bool
    input_tokens: int
    output_tokens: int


class CodeReviewService:
    def __init__(self) -> None:
        # MODEL_PATH peut être un dossier local ; sinon MODEL_NAME est utilisé.
        self.model_name_or_path = os.getenv("MODEL_PATH") or os.getenv(
            "MODEL_NAME", "microsoft/codereviewer"
        )
        self.local_files_only = env_flag("MODEL_LOCAL_FILES_ONLY", False)
        self.cache_dir = os.getenv("MODEL_CACHE_DIR") or None
        self.max_input_length = env_int("MAX_INPUT_LENGTH", 512)
        self.max_new_tokens = env_int("MAX_NEW_TOKENS", 96)
        self.num_beams = env_int("NUM_BEAMS", 4)
        self.device = resolve_device()

        self.tokenizer = None
        self.model = None

    @staticmethod
    def clean_generated_text(text: str) -> str:
        text = re.sub(r"<e\d+>", "", text)
        return text.strip()

    def _load_kwargs(self) -> Dict[str, object]:
        kwargs: Dict[str, object] = {"local_files_only": self.local_files_only}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        return kwargs

    def ensure_loaded(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return

        LOGGER.info(
            "Chargement du modèle '%s' (device=%s, local_files_only=%s)...",
            self.model_name_or_path,
            self.device,
            self.local_files_only,
        )

        load_kwargs = self._load_kwargs()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, **load_kwargs
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name_or_path, **load_kwargs
        )

        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(
            self.tokenizer, "eos_token", None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model.to(self.device)
        except RuntimeError as exc:
            if self.device.type == "cuda":
                LOGGER.warning(
                    "Impossible d'utiliser CUDA (%s). Basculage sur CPU.",
                    exc,
                )
                self.device = torch.device("cpu")
                self.model.to(self.device)
            else:
                raise

        self.model.eval()
        LOGGER.info("Modèle chargé avec succès.")

    @staticmethod
    def build_prompt(old_file: str, diff_hunk: str) -> str:
        """
        Prompt simple et lisible pour un prototype.
        Pour de meilleurs résultats, un fine-tuning sur la tâche msg reste préférable.
        """
        old_file = old_file.strip()
        diff_hunk = diff_hunk.strip()

        return (
            "old_file:\n"
            f"{old_file}\n\n"
            "diff_hunk:\n"
            f"{diff_hunk}\n\n"
            "review_comment:\n"
        )

    def _generate_once(self, prompt: str) -> GenerationResult:
        self.ensure_loaded()
        assert self.tokenizer is not None
        assert self.model is not None

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        truncated = int(encoded["input_ids"].shape[-1]) >= self.max_input_length
        input_tokens = int(encoded["input_ids"].shape[-1])

        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
        }

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id

        with torch.inference_mode():
            output_ids = self.model.generate(**encoded, **generation_kwargs)

        decoded = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        decoded = self.clean_generated_text(decoded)

        if not decoded:
            decoded = "Aucun commentaire n'a été généré."

        output_tokens = int(output_ids[0].shape[-1])
        return GenerationResult(
            text=decoded,
            truncated=truncated,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def generate_comment(self, old_file: str, diff_hunk: str) -> GenerationResult:
        prompt = self.build_prompt(old_file, diff_hunk)

        try:
            return self._generate_once(prompt)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device.type == "cuda":
                LOGGER.warning("Mémoire GPU insuffisante. Nouvelle tentative sur CPU.")
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                if self.model is not None:
                    self.model.to(self.device)
                return self._generate_once(prompt)
            raise


def create_app() -> Flask:
    app = Flask(__name__)
    service = CodeReviewService()

    # Sécurité minimale / configuration
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", secrets.token_hex(32))
    app.config["MAX_CONTENT_LENGTH"] = env_int("MAX_CONTENT_LENGTH", 64 * 1024)
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = env_flag("SESSION_COOKIE_SECURE", False)

    @app.route("/", methods=["GET", "POST"])
    def home():
        result = None
        old_file = ""
        diff_hunk = ""
        metadata = None

        if request.method == "POST":
            old_file = (request.form.get("old_file") or "").strip()
            diff_hunk = (request.form.get("diff_hunk") or "").strip()

            if not diff_hunk:
                flash("Le champ diff_hunk est obligatoire.", "error")
            elif not old_file:
                flash(
                    "Le champ old_file est obligatoire pour donner du contexte.",
                    "error",
                )
            elif len(old_file) > 20000 or len(diff_hunk) > 20000:
                flash(
                    "Entrée trop longue. Réduis la taille du code ou du diff.",
                    "error",
                )
            else:
                try:
                    generation = service.generate_comment(old_file, diff_hunk)
                    result = generation.text
                    metadata = {
                        "input_tokens": generation.input_tokens,
                        "output_tokens": generation.output_tokens,
                        "device": service.device.type,
                        "model_name": service.model_name_or_path,
                    }
                    flash("Analyse terminée avec succès.", "success")

                    if generation.truncated:
                        flash(
                            "Le prompt a été tronqué à la longueur maximale autorisée par le modèle.",
                            "warning",
                        )
                except Exception as exc:
                    LOGGER.exception("Erreur pendant la génération du commentaire.")
                    flash(f"Erreur d'inférence : {exc}", "error")

        return render_template(
            "index.html",
            result=result,
            old_file=old_file,
            diff_hunk=diff_hunk,
            metadata=metadata,
            model_name=service.model_name_or_path,
            device=service.device.type,
        )

    @app.route("/webhook", methods=["POST"])
    def github_webhook():
        token = os.getenv("GITHUB_TOKEN")
        webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET")

        if not token:
            LOGGER.error("GITHUB_TOKEN non configuré.")
            return {"error": "Server not configured"}, 500

        if webhook_secret:
            signature = request.headers.get("X-Hub-Signature-256", "")
            if not verify_signature(request.data, signature, webhook_secret):
                LOGGER.warning("Signature webhook invalide.")
                return {"error": "Invalid signature"}, 401

        event = request.headers.get("X-GitHub-Event", "")
        if event != "pull_request":
            return {"status": "ignored"}, 200

        payload = request.get_json(force=True)
        action = payload.get("action", "")
        if action not in ("opened", "synchronize", "reopened"):
            return {"status": "ignored"}, 200

        pr = payload["pull_request"]
        repo_full_name = payload["repository"]["full_name"]
        pr_number = pr["number"]
        base_sha = pr["base"]["sha"]

        LOGGER.info(
            "Traitement PR #%d sur %s (action=%s)", pr_number, repo_full_name, action
        )

        try:
            files = get_pr_files(repo_full_name, pr_number, token)
        except Exception as exc:
            LOGGER.exception("Impossible de récupérer les fichiers de la PR.")
            return {"error": str(exc)}, 500

        comments = []
        for f in files:
            patch = f.get("patch")
            if not patch:
                continue
            filename = f["filename"]
            old_content = (
                get_file_content(repo_full_name, filename, base_sha, token) or ""
            )
            try:
                result = service.generate_comment(old_content, patch)
                comments.append(f"**`{filename}`**\n{result.text}")
            except Exception as exc:
                LOGGER.warning("Échec de génération pour %s : %s", filename, exc)

        if not comments:
            return {"status": "no comments generated"}, 200

        body = "## Revue de code automatisée\n\n" + "\n\n---\n\n".join(comments)

        try:
            post_pr_comment(repo_full_name, pr_number, body, token)
        except Exception as exc:
            LOGGER.exception("Impossible de poster le commentaire.")
            return {"error": str(exc)}, 500

        return {"status": "ok", "comments": len(comments)}, 200

    @app.errorhandler(RequestEntityTooLarge)
    def handle_413(error):
        LOGGER.warning("Requête rejetée: taille supérieure à MAX_CONTENT_LENGTH.")
        return (
            render_template(
                "index.html",
                result=None,
                old_file="",
                diff_hunk="",
                metadata=None,
                model_name=service.model_name_or_path,
                device=service.device.type,
                page_error="Requête trop volumineuse. Réduis la taille du texte soumis.",
            ),
            413,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=env_int("FLASK_PORT", 5000),
        debug=env_flag("FLASK_DEBUG", False),
    )
