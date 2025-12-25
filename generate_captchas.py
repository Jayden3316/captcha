from __future__ import annotations

import argparse
import json
import random
import re
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import pandas as pd
from PIL.Image import new as createImage, Transform, Resampling
from PIL.ImageDraw import Draw, ImageDraw
from PIL.ImageFilter import SMOOTH
from PIL.ImageFont import FreeTypeFont, truetype


from src.generator import (
    random_color, 
    ConfigurableImageCaptcha, 
    random_capitalize
)

from src.utils import get_words, get_ttf_files, sanitize_alnum

from src.config.config import DatasetConfig

class CaptchaGenerator:
    def __init__(
        self,
        config: DatasetConfig,
        out_dir: str | Path,
        metadata_path: str | Path = "metadata.json",
        word_transform: Optional[Callable[[str], str]] = None,
    ):
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = Path(metadata_path)
        self.word_transform = word_transform  # e.g., lambda w: random_capitalize(w)
        
        self.captcha = ConfigurableImageCaptcha(
            width=config.width,
            height=config.target_height, # Use target_height for generation
            fonts=config.fonts,
            font_sizes=config.font_sizes,
            noise_bg_density=config.noise_bg_density,
            extra_spacing=config.extra_spacing,
            spacing_jitter=config.spacing_jitter,
            add_noise_dots=config.add_noise_dots,
            add_noise_curve=config.add_noise_curve,
            character_offset_dx=tuple(config.character_offset_dx) if config.character_offset_dx else (0, 0),
            character_offset_dy=tuple(config.character_offset_dy) if config.character_offset_dy else (0, 0),
            character_rotate=tuple(config.character_rotate) if config.character_rotate else (0, 0),
            character_warp_dx=tuple(config.character_warp_dx) if config.character_warp_dx else (0.1, 0.3),
            character_warp_dy=tuple(config.character_warp_dy) if config.character_warp_dy else (0.2, 0.3),
            word_space_probability=config.word_space_probability,
            word_offset_dx=config.word_offset_dx,
        )
        self.records: list[dict] = []

    def generate(self, words: Iterable[str]) -> None:
        """
        Generates captchas for each word, saves images, and writes metadata.json.
        Records the exact rendered word (after any transform) so case matches the image.
        """
        for word in words:
            clean = sanitize_alnum(word)
            if not clean:
                continue
            try:
                render_word = self.word_transform(clean) if self.word_transform else clean
                bg = tuple(self.config.bg_color) if self.config.bg_color else None
                fg = tuple(self.config.fg_color) if self.config.fg_color else None
                
                img = self.captcha.generate_image(
                    render_word,
                    bg_color=bg,
                    fg_color=fg,
                )
                # Use random suffix to prevent overwrite for repeated words
                import uuid
                filename = f"{render_word}_{uuid.uuid4().hex[:8]}.{self.config.image_ext}"
                fp = self.out_dir / filename
                img.save(fp)

                width, height = img.size
                self.records.append(
                    {
                        "image_path": str(fp.as_posix()),
                        "word_input": clean,  # original supplied word
                        "word_rendered": render_word,  # exact casing used in the image
                        "word_length": len(render_word),
                        "width": width,
                        "height": height,
                    }
                )
            except Exception as e:
                print(f"Failed to render {word}: {e}")

        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.records)} entries to {self.metadata_path}")





