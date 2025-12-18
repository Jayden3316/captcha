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
from captcha.image import ImageCaptcha
from PIL.Image import new as createImage
from PIL.ImageDraw import Draw


def sanitize_alnum(text: str) -> str:
    """
    Keep only alphanumeric characters (A–Z, a–z, 0–9) in the string.
    """
    return re.sub(r"[^0-9A-Za-z]", "", text)


def get_words(
    file_path: str,
    min_word_len: int = 4,
    max_word_len: Optional[int] = None,
) -> list[str]:
    """
    Load words from a TSV file and filter by length and alnum content.
    """
    df = pd.read_csv(file_path, sep="\t", names=["word_id", "word", "frequency"])
    words: set[str] = set()
    for word in df["word"].tolist():
        for split_word in word.split():
            clean = sanitize_alnum(split_word)
            n = len(clean)
            if n >= min_word_len and (max_word_len is None or n <= max_word_len):
                words.add(clean)
    return sorted(words)


class NoisySpacedImageCaptcha(ImageCaptcha):
    def __init__(
        self,
        width: int = 200,
        height: int = 70,
        fonts=None,
        font_sizes=None,
        noise_bg_density: int = 5000,
        extra_spacing: int = -5,
        spacing_jitter: int = 6,
    ):
        super().__init__(width=width, height=height, fonts=fonts, font_sizes=font_sizes)
        self.noise_bg_density = noise_bg_density
        self.extra_spacing = extra_spacing
        self.spacing_jitter = spacing_jitter

    def _add_background_noise(self, image):
        draw = Draw(image)
        w, h = image.size
        for _ in range(self.noise_bg_density):
            x, y = secrets.randbelow(w), secrets.randbelow(h)
            val = secrets.randbelow(120) + 80  # soft gray-ish
            draw.point((x, y), fill=(val, val, val))
        return image

    def create_captcha_image(self, chars, color, background):
        temp = createImage("RGB", (self._width, self._height), background)
        draw = Draw(temp)

        images = []
        for c in chars:
            if secrets.randbits(32) / (2**32) > self.word_space_probability:
                images.append(self._draw_character(" ", draw, color))
            images.append(self._draw_character(c, draw, color))

        text_width = sum(im.size[0] for im in images)
        average = int(text_width / max(len(chars), 1))
        rand = int(self.word_offset_dx * average)
        pad = 16

        per_gap_max = self.extra_spacing + max(self.spacing_jitter, 0) + rand
        dyn_width = max(self._width, text_width + len(images) * per_gap_max + pad)

        image = createImage("RGB", (dyn_width, self._height), background)
        draw = Draw(image)

        offset = pad // 2
        for im in images:
            w, h = im.size
            mask = im.convert("L").point(self.lookup_table)
            image.paste(im, (offset, int((self._height - h) / 2)), mask)
            jitter = random.randint(-self.spacing_jitter, self.spacing_jitter)
            step = w + self.extra_spacing + max(jitter, 0) + rand
            offset += step

        self._add_background_noise(image)
        return image


class CaptchaGenerator:
    def __init__(
        self,
        out_dir: str | Path,
        metadata_path: str | Path = "metadata.json",
        width: int = 200,
        height: int = 70,
        fonts=None,
        font_sizes=None,
        noise_bg_density: int = 5000,
        extra_spacing: int = -5,
        spacing_jitter: int = 6,
        bg_color: Optional[tuple[int, int, int]] = None,
        fg_color: Optional[tuple[int, int, int, int]] = None,
        image_ext: str = "png",
        word_transform: Optional[Callable[[str], str]] = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = Path(metadata_path)
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.image_ext = image_ext.lower().lstrip(".")
        self.word_transform = word_transform  # e.g., lambda w: random_capitalize(w)
        self.captcha = NoisySpacedImageCaptcha(
            width=width,
            height=height,
            fonts=fonts,
            font_sizes=font_sizes,
            noise_bg_density=noise_bg_density,
            extra_spacing=extra_spacing,
            spacing_jitter=spacing_jitter,
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
                img = self.captcha.generate_image(
                    render_word,
                    bg_color=self.bg_color,
                    fg_color=self.fg_color,
                )
                filename = f"{render_word}.{self.image_ext}"
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
            except Exception:
                print(f"Failed to render {word}")

        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.records)} entries to {self.metadata_path}")


def random_capitalize(s: str) -> str:
    return "".join(c.upper() if random.random() < 0.5 else c.lower() for c in s)


def get_ttf_files(root_path: str | Path) -> List[str]:
    """
    Recursively find all .ttf files in the given directory tree.

    Args:
        root_path: Path to the root directory to search (e.g., 'font_library')

    Returns:
        A list of string paths for all .ttf files found
    """
    root = Path(root_path)
    ttf_files: List[str] = []

    for file_path in root.rglob("*.ttf"):
        ttf_files.append(str(file_path))

    return sorted(ttf_files)
