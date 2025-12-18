from __future__ import annotations

import argparse
import json
import random
import re
import secrets
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import torch
from generate_captchas import (
    get_ttf_files, 
    sanitize_alnum, 
    NoisySpacedImageCaptcha, 
    CaptchaGenerator,
    random_capitalize,
    get_words
)
from train import train

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Captcha Tool: Generate, Train, Evaluate."
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # --- Generate Command ---
    gen_parser = subparsers.add_parser("generate", help="Generate captcha dataset")
    # Paths
    gen_parser.add_argument("--word-file", type=str, required=True, help="Path to TSV word list")
    gen_parser.add_argument("--font-root", type=str, required=True, help="Root directory for fonts")
    gen_parser.add_argument("--out-dir", type=str, default="../validation_set", help="Output directory")
    gen_parser.add_argument("--metadata-path", type=str, default="metadata.json", help="Metadata JSON path")
    # Params
    gen_parser.add_argument("--min-word-len", type=int, default=4)
    gen_parser.add_argument("--max-word-len", type=int, default=None)
    gen_parser.add_argument("--width", type=int, default=200)
    gen_parser.add_argument("--height", type=int, default=70)
    gen_parser.add_argument("--noise-bg-density", type=int, default=5000)
    gen_parser.add_argument("--extra-spacing", type=int, default=-5)
    gen_parser.add_argument("--spacing-jitter", type=int, default=6)
    gen_parser.add_argument("--bg-color", type=str, default=None)
    gen_parser.add_argument("--fg-color", type=str, default=None)
    gen_parser.add_argument("--image-ext", type=str, default="png")
    gen_parser.add_argument("--max-fonts-per-family", type=int, default=2)
    gen_parser.add_argument("--no-random-capitalize", action="store_true")

    # --- Train Command ---
    train_parser = subparsers.add_parser("train", help="Train the Captcha Model")
    train_parser.add_argument("--metadata-path", type=str, default="validation_set/metadata.json")
    train_parser.add_argument("--image-base-dir", type=str, default=".")
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    train_parser.add_argument("--wandb-project", type=str, default="captcha-ocr")

    return parser

def main(args: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)

    if parsed.command == "generate":
        # Collect fonts
        all_fonts = get_ttf_files(parsed.font_root)
        family_seen: dict[str, int] = defaultdict(int)
        selected_fonts: list[str] = []

        for p in all_fonts:
            family = Path(p).parent.name
            if family_seen[family] < parsed.max_fonts_per_family:
                selected_fonts.append(p)
                family_seen[family] += 1

        words = get_words(
            parsed.word_file,
            min_word_len=parsed.min_word_len,
            max_word_len=parsed.max_word_len,
        )

        transform: Optional[Callable[[str], str]] = None if parsed.no_random_capitalize else random_capitalize

        gen = CaptchaGenerator(
            out_dir=parsed.out_dir,
            metadata_path=parsed.metadata_path,
            width=parsed.width,
            height=parsed.height,
            fonts=selected_fonts,
            noise_bg_density=parsed.noise_bg_density,
            extra_spacing=parsed.extra_spacing,
            spacing_jitter=parsed.spacing_jitter,
            bg_color=parsed.bg_color,
            fg_color=parsed.fg_color,
            image_ext=parsed.image_ext,
            word_transform=transform,
        )
        gen.generate(words)

    elif parsed.command == "train":
        train(
            metadata_path=parsed.metadata_path,
            image_base_dir=parsed.image_base_dir,
            batch_size=parsed.batch_size,
            epochs=parsed.epochs,
            lr=parsed.lr,
            checkpoint_dir=parsed.checkpoint_dir,
            wandb_project=parsed.wandb_project
        )

if __name__ == "__main__":
    main()