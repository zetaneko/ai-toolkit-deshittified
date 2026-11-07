import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


def _load_slider_prompts():
    path = (
        Path(__file__).resolve().parents[1]
        / "extensions_built_in"
        / "concept_slider"
        / "slider_prompts.py"
    )
    spec = importlib.util.spec_from_file_location("slider_prompts_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


slider_prompts = _load_slider_prompts()


class _Config:
    def __init__(self, caption_ext=".txt", default_caption=None):
        self.caption_ext = caption_ext
        self.default_caption = default_caption


class SliderPromptTests(unittest.TestCase):
    def test_no_sidecar_uses_job_prompts(self):
        with tempfile.TemporaryDirectory() as folder:
            image = os.path.join(folder, "front.jpg")
            open(image, "w").close()
            positive, negative = slider_prompts.read_slider_prompt_overrides(image)
            self.assertIsNone(positive)
            self.assertIsNone(negative)
            resolved = slider_prompts.resolve_slider_prompts(
                "smiling",
                "frowning",
                raw_caption="person facing camera",
                positive_override=positive,
                negative_override=negative,
            )
            self.assertEqual(resolved, ("smiling", "frowning"))

    def test_slider_json_owns_both_sides(self):
        with tempfile.TemporaryDirectory() as folder:
            image = os.path.join(folder, "left.png")
            open(image, "w").close()
            stem = os.path.splitext(image)[0]
            with open(stem + ".slider.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {"positive_prompt": "left profile, open smile", "negative_prompt": ""},
                    handle,
                )
            with open(stem + ".pos.txt", "w", encoding="utf-8") as handle:
                handle.write("ignored positive")
            with open(stem + ".neg.txt", "w", encoding="utf-8") as handle:
                handle.write("ignored negative")
            positive, negative = slider_prompts.read_slider_prompt_overrides(image)
            self.assertEqual(positive, "left profile, open smile")
            self.assertIsNone(negative)
            resolved = slider_prompts.resolve_slider_prompts(
                "job positive",
                "job negative",
                positive_override=positive,
                negative_override=negative,
            )
            self.assertEqual(resolved, ("left profile, open smile", "job negative"))

    def test_text_sidecars_when_json_is_absent(self):
        with tempfile.TemporaryDirectory() as folder:
            image = os.path.join(folder, "right.webp")
            open(image, "w").close()
            stem = os.path.splitext(image)[0]
            with open(stem + ".neg.txt", "w", encoding="utf-8") as handle:
                handle.write("  right profile, frown \n")
            positive, negative = slider_prompts.read_slider_prompt_overrides(image)
            self.assertIsNone(positive)
            self.assertEqual(negative, "right profile, frown")

    def test_json_aliases_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            image = os.path.join(folder, "a.jpg")
            open(image, "w").close()
            stem = os.path.splitext(image)[0]
            with open(stem + ".slider.json", "w", encoding="utf-8") as handle:
                json.dump({"positive": "alias pos", "negative": "alias neg"}, handle)
            self.assertEqual(
                slider_prompts.read_slider_prompt_overrides(image),
                ("alias pos", "alias neg"),
            )
            with open(stem + ".slider.json", "w", encoding="utf-8") as handle:
                handle.write("{")
            with self.assertRaises(ValueError):
                slider_prompts.read_slider_prompt_overrides(image)

    def test_caption_and_trigger_placeholders(self):
        resolved = slider_prompts.resolve_slider_prompts(
            "[caption], smiling",
            "{caption}, frowning [trigger]",
            raw_caption="person in left profile",
            trigger="ohwx",
        )
        self.assertEqual(
            resolved,
            (
                "person in left profile, smiling",
                "person in left profile, frowning ohwx",
            ),
        )

    def test_trigger_token_stays_when_unset(self):
        resolved = slider_prompts.resolve_slider_prompts(
            "[trigger] smiling",
            "frowning",
            trigger="  ",
        )
        self.assertEqual(resolved, ("[trigger] smiling", "frowning"))

    def test_read_dataset_caption_file_default_and_dict(self):
        with tempfile.TemporaryDirectory() as folder:
            image = os.path.join(folder, "shot.jpg")
            open(image, "w").close()
            stem = os.path.splitext(image)[0]
            with open(stem + ".txt", "w", encoding="utf-8") as handle:
                handle.write(" three quarter view \n")
            self.assertEqual(
                slider_prompts.read_dataset_caption(image, _Config()),
                "three quarter view",
            )
            self.assertEqual(
                slider_prompts.read_dataset_caption(
                    image,
                    _Config(caption_ext="json"),
                    {image: {"caption": "from dict"}},
                ),
                "from dict",
            )
            missing = os.path.join(folder, "empty.jpg")
            open(missing, "w").close()
            self.assertEqual(
                slider_prompts.read_dataset_caption(missing, _Config(default_caption="person")),
                "person",
            )


if __name__ == "__main__":
    unittest.main()
