"""Per-image positive and negative prompts for concept sliders.

The job-level slider prompts still apply to every image. An image can override
either side. This is the path used for Krea 2 concept sliders when the wording
needs to change with the angle of the subject.

Next to ``photo.jpg``, in this order:

- ``photo.slider.json`` owns both sides. Blank or missing fields fall back to
  the job prompt. ``.pos.txt`` / ``.neg.txt`` are ignored while this file exists.
- otherwise ``photo.pos.txt`` and ``photo.neg.txt`` override one side each.
  A side with no text file keeps looking.
- otherwise ``slider_prompts.json`` in the same folder. It is one object with
  an ``images`` list. An entry matches the image by its ``image`` filename.
  ``positive`` / ``positive_prompt`` and ``negative`` / ``negative_prompt``
  are the full prompts for that image. A sidecar still wins over this file.

A prompt may contain ``[caption]`` (or ``{caption}``). That token is replaced
with the image caption. ``[trigger]`` is replaced when the dataset has a
trigger word.
"""

import json
import os
from typing import Optional


def sidecar_paths(image_path: str) -> tuple[str, str, str]:
    stem = os.path.splitext(image_path)[0]
    return stem + ".slider.json", stem + ".pos.txt", stem + ".neg.txt"


def _blank_to_none(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text


def _read_text_override(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return _blank_to_none(handle.read())


def _pick_field(data: dict, primary: str, alias: str) -> Optional[str]:
    if primary in data:
        return _blank_to_none(data.get(primary))
    if alias in data:
        return _blank_to_none(data.get(alias))
    return None


# Parsed folder maps, keyed by absolute directory. Training reads each image
# once before the text encoder is unloaded, so this stays for the process.
_folder_prompt_cache: dict[str, dict[str, tuple[Optional[str], Optional[str]]]] = {}


def _load_folder_prompt_map(directory: str) -> dict[str, tuple[Optional[str], Optional[str]]]:
    directory = os.path.abspath(directory)
    cached = _folder_prompt_cache.get(directory)
    if cached is not None:
        return cached

    path = os.path.join(directory, "slider_prompts.json")
    mapping: dict[str, tuple[Optional[str], Optional[str]]] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid concept slider prompt file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Concept slider prompt file {path} must be a JSON object")
        images = data.get("images", [])
        if not isinstance(images, list):
            raise ValueError(
                f"Concept slider prompt file {path} field 'images' must be a list"
            )
        for entry in images:
            if not isinstance(entry, dict):
                continue
            name = entry.get("image") or entry.get("filename") or entry.get("file")
            if not name:
                continue
            key = os.path.basename(str(name)).strip().lower()
            if key == "":
                continue
            mapping[key] = (
                _pick_field(entry, "positive_prompt", "positive"),
                _pick_field(entry, "negative_prompt", "negative"),
            )
    _folder_prompt_cache[directory] = mapping
    return mapping


def read_slider_prompt_overrides(image_path: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(positive, negative)``. ``None`` means that side has no override."""
    json_path, pos_path, neg_path = sidecar_paths(image_path)
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid concept slider prompt file {json_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Concept slider prompt file {json_path} must be a JSON object")
        return (
            _pick_field(data, "positive_prompt", "positive"),
            _pick_field(data, "negative_prompt", "negative"),
        )

    positive = _read_text_override(pos_path)
    negative = _read_text_override(neg_path)
    if positive is None or negative is None:
        folder_map = _load_folder_prompt_map(os.path.dirname(image_path) or ".")
        folder_positive, folder_negative = folder_map.get(
            os.path.basename(image_path).lower(),
            (None, None),
        )
        if positive is None:
            positive = folder_positive
        if negative is None:
            negative = folder_negative
    return positive, negative


def apply_prompt_placeholders(
    prompt: str,
    caption: str,
    trigger: Optional[str] = None,
) -> str:
    text = prompt or ""
    caption_text = caption or ""
    text = text.replace("[caption]", caption_text).replace("{caption}", caption_text)
    trigger_text = (trigger or "").strip()
    if trigger_text:
        text = text.replace("[trigger]", trigger_text)
    return text


def resolve_slider_prompts(
    global_positive: str,
    global_negative: str,
    raw_caption: str = "",
    trigger: Optional[str] = None,
    positive_override: Optional[str] = None,
    negative_override: Optional[str] = None,
) -> tuple[str, str]:
    """Pick the prompt for one image, then fill ``[caption]`` and ``[trigger]``."""
    positive = positive_override if positive_override is not None else (global_positive or "")
    negative = negative_override if negative_override is not None else (global_negative or "")
    return (
        apply_prompt_placeholders(positive, raw_caption, trigger),
        apply_prompt_placeholders(negative, raw_caption, trigger),
    )


def read_dataset_caption(image_path: str, dataset_config, caption_dict=None) -> str:
    """Caption text used for ``[caption]``. Does not apply dropout or token shuffle."""
    if caption_dict is not None and image_path in caption_dict:
        entry = caption_dict[image_path]
        if isinstance(entry, dict) and entry.get("caption"):
            return str(entry["caption"]).strip()
        if isinstance(entry, str) and entry.strip():
            return entry.strip()

    ext = getattr(dataset_config, "caption_ext", None) or ".txt"
    if not str(ext).startswith("."):
        ext = "." + str(ext)
    prompt_path = os.path.splitext(image_path)[0] + ext
    prompt = ""
    if os.path.isfile(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as handle:
            prompt = handle.read().strip()
    if prompt == "":
        default_caption = getattr(dataset_config, "default_caption", None)
        if default_caption:
            prompt = str(default_caption).strip()
    return prompt
