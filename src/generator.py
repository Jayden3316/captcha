import secrets
import random
import re
from typing import List, Optional
from PIL import Image, ImageDraw
from PIL.Image import new as createImage, Transform, Resampling
from PIL.ImageDraw import Draw
from PIL.ImageFilter import SMOOTH
from PIL.ImageFont import FreeTypeFont, truetype


def sanitize_alnum(text: str) -> str:
    """
    Keep only alphanumeric characters (A–Z, a–z, 0–9) in the string.
    """
    return re.sub(r"[^0-9A-Za-z]", "", text)


def random_color(start: int, end: int, opacity: Optional[int] = None) -> tuple:
    red = secrets.randbelow(end - start + 1) + start
    green = secrets.randbelow(end - start + 1) + start
    blue = secrets.randbelow(end - start + 1) + start
    if opacity is None:
        return red, green, blue
    return red, green, blue, opacity


def random_capitalize(s: str) -> str:
    return "".join(c.upper() if random.random() < 0.5 else c.lower() for c in s)


class ConfigurableImageCaptcha:
    """
    Fully configurable ImageCaptcha implementation.
    Allows fine-grained control over distortions, noise, and spacing.
    """
    def __init__(
        self,
        width: int = 200,
        height: int = 70,
        fonts: List[str] | None = None,
        font_sizes: tuple[int, ...] | None = None,
        noise_bg_density: int = 5000,
        extra_spacing: int = -5,
        spacing_jitter: int = 6,
        add_noise_dots: bool = True,
        add_noise_curve: bool = True,
        # Fine-grained params
        character_offset_dx: tuple[int, int] = (0, 4),
        character_offset_dy: tuple[int, int] = (0, 6),
        character_rotate: tuple[int, int] = (-30, 30),
        character_warp_dx: tuple[float, float] = (0.1, 0.3),
        character_warp_dy: tuple[float, float] = (0.2, 0.3),
        word_space_probability: float = 0.5,
        word_offset_dx: float = 0.25,
    ):
        self._width = width
        self._height = height
        self._fonts = fonts or []
        self._font_sizes = font_sizes or (42, 50, 56)
        self._truefonts: List[FreeTypeFont] = []
        
        # Instance-level configuration for strict control
        self.noise_bg_density = noise_bg_density
        self.extra_spacing = extra_spacing
        self.spacing_jitter = spacing_jitter
        self.add_noise_dots = add_noise_dots
        self.add_noise_curve = add_noise_curve
        
        # Distortions (Tuples enforced at call site, but good to be safe)
        self.character_offset_dx = character_offset_dx
        self.character_offset_dy = character_offset_dy
        self.character_rotate = character_rotate
        self.character_warp_dx = character_warp_dx
        self.character_warp_dy = character_warp_dy
        self.word_space_probability = word_space_probability
        self.word_offset_dx = word_offset_dx
        
        self.lookup_table: list[int] = [int(i * 1.97) for i in range(256)]

    @property
    def truefonts(self) -> List[FreeTypeFont]:
        if self._truefonts:
            return self._truefonts
        # Try loading fonts, handle potential errors if paths are bad
        try:
             self._truefonts = [
                truetype(n, s)
                for n in self._fonts
                for s in self._font_sizes
            ]
        except Exception as e:
            print(f"Warning: Failed to load some fonts: {e}")
            # Fallback? No, just crash or let it be empty if really broken, but we expect valid paths
        return self._truefonts

    def _draw_character(self, c: str, draw: ImageDraw, color: tuple) -> tuple[Image.Image, int]:
        font = secrets.choice(self.truefonts)
        
        # bbox = (left, top, right, bottom)
        bbox = draw.textbbox((0, 0), c, font=font, anchor="ls")
        left, top, right, bottom = bbox
        
        w = right - left
        h = bottom - top
        
        # Apply random offsets first
        dx = secrets.randbelow(self.character_offset_dx[1] - self.character_offset_dx[0] + 1) + self.character_offset_dx[0]
        dy = secrets.randbelow(self.character_offset_dy[1] - self.character_offset_dy[0] + 1) + self.character_offset_dy[0]
        
        # New width/height with offsets
        canvas_w = w + abs(dx) + 10 # extra padding
        canvas_h = h + abs(dy) + 10
        
        im = createImage('RGBA', (canvas_w, canvas_h))
        local_draw = Draw(im)
        
        draw_x = -left + dx 
        draw_y = -top + dy  
        
        local_draw.text((draw_x, draw_y), c, font=font, fill=color, anchor="ls")

        # Now we crop
        cropped_bbox = im.getbbox()
        if cropped_bbox:
            im = im.crop(cropped_bbox)
            crop_left, crop_top, _, _ = cropped_bbox
            baseline_offset = draw_y - crop_top
        else:
            baseline_offset = 0

        # Rotate
        if self.character_rotate != (0, 0):
             angle = self.character_rotate[0] + (secrets.randbits(32) / (2**32)) * (self.character_rotate[1] - self.character_rotate[0])
             im = im.rotate(angle, Resampling.BILINEAR, expand=True)

        # Warp
        dx2 = im.size[0] * (secrets.randbits(32) / (2**32)) * (self.character_warp_dx[1] - self.character_warp_dx[0]) + self.character_warp_dx[0]
        dy2 = im.size[1] * (secrets.randbits(32) / (2**32)) * (self.character_warp_dy[1] - self.character_warp_dy[0]) + self.character_warp_dy[0]
        
        if dx2 != 0 or dy2 != 0:
            w, h = im.size
            x1 = int(secrets.randbits(32) / (2**32) * (dx2 - (-dx2)) + (-dx2))
            y1 = int(secrets.randbits(32) / (2**32) * (dy2 - (-dy2)) + (-dy2))
            x2 = int(secrets.randbits(32) / (2**32) * (dx2 - (-dx2)) + (-dx2))
            y2 = int(secrets.randbits(32) / (2**32) * (dy2 - (-dy2)) + (-dy2))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1, y1,
                -x1, h2 - y2,
                w2 + x2, h2 + y2,
                w2 - x2, -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((int(w), int(h)), Transform.QUAD, data)
            
        return im, int(baseline_offset)

    def _add_background_noise(self, image):
        if self.noise_bg_density <= 0:
            return image
        draw = Draw(image)
        w, h = image.size
        for _ in range(self.noise_bg_density):
            x, y = secrets.randbelow(w), secrets.randbelow(h)
            val = secrets.randbelow(120) + 80  
            draw.point((x, y), fill=(val, val, val))
        return image
    
    def create_noise_dots(self, image, color, width=3, number=30):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = secrets.randbelow(w + 1)
            y1 = secrets.randbelow(h + 1)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image

    def create_noise_curve(self, image, color):
        w, h = image.size
        x1 = secrets.randbelow(int(w / 5) + 1)
        x2 = secrets.randbelow(w - int(w / 5) + 1) + int(w / 5)
        y1 = secrets.randbelow(h - 2 * int(h / 5) + 1) + int(h / 5)
        y2 = secrets.randbelow(h - y1 - int(h / 5) + 1) + y1
        points = [x1, y1, x2, y2]
        end = secrets.randbelow(41) + 160
        start = secrets.randbelow(21)
        Draw(image).arc(points, start, end, fill=color)
        return image

    def create_captcha_image(self, chars, color, background):
        temp = createImage("RGB", (self._width, self._height), background)
        draw = Draw(temp)

        images = []
        for c in chars:
            if secrets.randbits(32) / (2**32) < self.word_space_probability:
                sp_im, _ = self._draw_character(" ", draw, color)
                images.append((sp_im, 0))
            images.append(self._draw_character(c, draw, color))

        text_width = sum(im.size[0] for im, _ in images)
        average = int(text_width / max(len(chars), 1))
        
        # Word offset random
        rand = int(self.word_offset_dx * average) if average else 0
        pad = 16

        per_gap_max = self.extra_spacing + max(self.spacing_jitter, 0) + rand
        dyn_width = max(self._width, text_width + len(images) * per_gap_max + pad)

        image = createImage("RGB", (dyn_width, self._height), background)
        draw = Draw(image)
        
        # Consistent baseline at 75% of height
        baseline_y = int(self._height * 0.75) 

        offset = pad // 2
        for im, ascent_offset in images:
            w, h = im.size
            if im.mode == "RGBA":
                mask = im.split()[3]
            else:
                mask = im.convert("L").point(self.lookup_table)
            
            y_pos = baseline_y - ascent_offset
            
            image.paste(im, (offset, y_pos), mask)
            
            jitter = random.randint(-self.spacing_jitter, self.spacing_jitter) if self.spacing_jitter > 0 else 0
            step = w + self.extra_spacing + max(jitter, 0) + rand
            offset += step

        self._add_background_noise(image)
        return image

    def generate_image(self, chars: str, bg_color=None, fg_color=None):
        background = bg_color if bg_color else random_color(238, 255)
        random_fg_color = random_color(10, 200, secrets.randbelow(36) + 220)
        color = fg_color if fg_color else random_fg_color

        im = self.create_captcha_image(chars, color, background)

        if self.add_noise_dots:
            self.create_noise_dots(im, color)
        
        if self.add_noise_curve:
            self.create_noise_curve(im, color)

        im = im.filter(SMOOTH)
        return im
