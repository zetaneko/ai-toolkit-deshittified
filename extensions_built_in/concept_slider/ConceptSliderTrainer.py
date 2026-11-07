import os
from collections import OrderedDict
from typing import Optional

import torch

from extensions_built_in.concept_slider.slider_prompts import (
    read_dataset_caption,
    read_slider_prompt_overrides,
    resolve_slider_prompts,
)
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.data_loader import get_dataloader_datasets
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.train_tools import get_torch_dtype


class ConceptSliderTrainerConfig:
    def __init__(self, **kwargs):
        self.guidance_strength: float = kwargs.get("guidance_strength", 3.0)
        self.anchor_strength: float = kwargs.get("anchor_strength", 1.0)
        self.positive_prompt: str = kwargs.get("positive_prompt", "")
        self.negative_prompt: str = kwargs.get("negative_prompt", "")
        self.target_class: str = kwargs.get("target_class", "")
        self.anchor_class: Optional[str] = kwargs.get("anchor_class", None)
        # When true, each dataset image may supply its own positive and negative
        # prompt. Images without one keep the job prompts above. See slider_prompts.py.
        self.per_image_prompts: bool = _as_bool(kwargs.get("per_image_prompts", False))


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _short_prompt(prompt: str, limit: int = 160) -> str:
    text = " ".join((prompt or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def norm_like_tensor(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalize the tensor to have the same mean and std as the target tensor."""
    tensor_mean = tensor.mean()
    tensor_std = tensor.std()
    target_mean = target.mean()
    target_std = target.std()
    normalized_tensor = (tensor - tensor_mean) / (
        tensor_std + 1e-8
    ) * target_std + target_mean
    return normalized_tensor


class ConceptSliderTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.do_guided_loss = True

        self.slider: ConceptSliderTrainerConfig = ConceptSliderTrainerConfig(
            **self.config.get("slider", {})
        )

        self.positive_prompt = self.slider.positive_prompt
        self.positive_prompt_embeds: Optional[PromptEmbeds] = None
        self.negative_prompt = self.slider.negative_prompt
        self.negative_prompt_embeds: Optional[PromptEmbeds] = None
        self.target_class = self.slider.target_class
        self.target_class_embeds: Optional[PromptEmbeds] = None
        self.anchor_class = self.slider.anchor_class
        self.anchor_class_embeds: Optional[PromptEmbeds] = None
        # CPU copies keyed by the exact prompt string. PromptEmbeds.to mutates,
        # so the batch path clones out of this cache before moving to the device.
        self._prompt_embed_cache: dict[str, PromptEmbeds] = {}
        self._image_prompt_pairs: Optional[dict[str, tuple[str, str]]] = None
        self._use_per_image_prompt_batches = False

    def _cache_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        cached = self._prompt_embed_cache.get(prompt)
        if cached is not None:
            return cached
        embeds = self.sd.encode_prompt([prompt]).detach()
        stored = embeds.clone().to("cpu")
        self._prompt_embed_cache[prompt] = stored
        return stored

    def _encode_slider_prompt(self, prompt: str) -> PromptEmbeds:
        """Encode once, keep a CPU copy, and return a device copy for the fast path."""
        cached = self._cache_prompt_embeds(prompt)
        return cached.clone().to(self.device_torch, dtype=self.sd.torch_dtype)

    def _iter_dataset_file_items(self):
        if self.data_loader is None:
            return
        for dataset in get_dataloader_datasets(self.data_loader):
            file_list = getattr(dataset, "file_list", None)
            if not file_list:
                continue
            caption_dict = getattr(dataset, "caption_dict", None)
            for file_item in file_list:
                yield file_item, caption_dict

    def _prepare_per_image_prompts(self):
        pairs: dict[str, tuple[str, str]] = {}
        examples: list[tuple[str, str, str]] = []
        seen_paths = set()
        for file_item, caption_dict in self._iter_dataset_file_items():
            path = file_item.path
            if path in seen_paths:
                continue
            seen_paths.add(path)
            caption = read_dataset_caption(path, file_item.dataset_config, caption_dict)
            positive_override, negative_override = read_slider_prompt_overrides(path)
            positive, negative = resolve_slider_prompts(
                global_positive=self.positive_prompt,
                global_negative=self.negative_prompt,
                raw_caption=caption,
                trigger=getattr(file_item, "trigger_word", None),
                positive_override=positive_override,
                negative_override=negative_override,
            )
            pairs[path] = (positive, negative)
            if positive != self.positive_prompt or negative != self.negative_prompt:
                if len(examples) < 3:
                    examples.append((os.path.basename(path), positive, negative))

        self._image_prompt_pairs = pairs
        custom = sum(
            1
            for positive, negative in pairs.values()
            if positive != self.positive_prompt or negative != self.negative_prompt
        )
        self._use_per_image_prompt_batches = custom > 0
        if custom == 0:
            print_acc(
                "Per-image concept slider prompts are on, but every image uses the job prompts."
            )
            return

        unique_prompts = {prompt for pair in pairs.values() for prompt in pair}
        for prompt in unique_prompts:
            self._cache_prompt_embeds(prompt)
        print_acc(
            f"Per-image concept slider prompts: {custom} of {len(pairs)} images "
            f"differ from the job prompts ({len(unique_prompts)} unique prompt strings cached)."
        )
        long_prompts = [prompt for prompt in unique_prompts if len(prompt) > 2000]
        if long_prompts:
            limit = getattr(self.sd, "max_text_length", None)
            print_acc(
                f"  {len(long_prompts)} of those prompts are over 2000 characters. "
                f"Krea 2 truncates each prompt at max_text_length"
                + (f" ({limit} tokens)." if limit else " (default 512 tokens).")
            )
        for name, positive, negative in examples:
            print_acc(f"  {name}")
            print_acc(f"    positive: {_short_prompt(positive)}")
            print_acc(f"    negative: {_short_prompt(negative)}")

    def _prompt_pair_for(self, file_item) -> tuple[str, str]:
        if self._image_prompt_pairs is not None:
            pair = self._image_prompt_pairs.get(file_item.path)
            if pair is not None:
                return pair
        return self.positive_prompt, self.negative_prompt

    def _stack_cached_prompts(self, prompts: list[str], dtype) -> PromptEmbeds:
        pieces = []
        for prompt in prompts:
            cached = self._prompt_embed_cache.get(prompt)
            if cached is None:
                raise RuntimeError(
                    "Concept slider prompt was not encoded before the text encoder was unloaded: "
                    + repr(prompt)
                )
            pieces.append(cached.clone())
        return concat_prompt_embeds(pieces).to(self.device_torch, dtype=dtype)

    def hook_before_train_loop(self):
        # do this before calling parent as it unloads the text encoder if requested
        if self.is_caching_text_embeddings:
            # make sure model is on cpu for this part so we don't oom.
            self.sd.unet.to("cpu")

        # cache the job-level slider prompts. Per-image overrides are encoded
        # here too, while the text encoder is still loaded.
        with torch.no_grad():
            self.positive_prompt_embeds = self._encode_slider_prompt(self.positive_prompt)
            self.target_class_embeds = self._encode_slider_prompt(self.target_class)
            self.negative_prompt_embeds = self._encode_slider_prompt(self.negative_prompt)

            if self.anchor_class is not None:
                self.anchor_class_embeds = self._encode_slider_prompt(self.anchor_class)

            if self.slider.per_image_prompts:
                self._prepare_per_image_prompts()

        # call parent
        super().hook_before_train_loop()

    def get_guided_loss(
        self,
        noisy_latents: torch.Tensor,
        conditional_embeds: PromptEmbeds,
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: "DataLoaderBatchDTO",
        noise: torch.Tensor,
        unconditional_embeds: Optional[PromptEmbeds] = None,
        **kwargs,
    ):
        # todo for embeddings, we need to run without trigger words
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False

        # do out prior preds first
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)
            self.sd.unet.eval()
            noisy_latents = noisy_latents.to(self.device_torch, dtype=dtype).detach()

            batch_size = noisy_latents.shape[0]

            if self._use_per_image_prompt_batches:
                if len(batch.file_items) != batch_size:
                    raise ValueError(
                        "Per-image concept slider prompts need one prompt pair per latent "
                        f"({len(batch.file_items)} file items, batch size {batch_size})."
                    )
                positive_embeds = self._stack_cached_prompts(
                    [self._prompt_pair_for(file_item)[0] for file_item in batch.file_items],
                    dtype,
                )
                negative_embeds = self._stack_cached_prompts(
                    [self._prompt_pair_for(file_item)[1] for file_item in batch.file_items],
                    dtype,
                )
            else:
                positive_embeds = concat_prompt_embeds(
                    [self.positive_prompt_embeds] * batch_size
                ).to(self.device_torch, dtype=dtype)
                negative_embeds = concat_prompt_embeds(
                    [self.negative_prompt_embeds] * batch_size
                ).to(self.device_torch, dtype=dtype)
            target_class_embeds = concat_prompt_embeds(
                [self.target_class_embeds] * batch_size
            ).to(self.device_torch, dtype=dtype)

            if self.anchor_class_embeds is not None:
                anchor_embeds = concat_prompt_embeds(
                    [self.anchor_class_embeds] * batch_size
                ).to(self.device_torch, dtype=dtype)

            if self.anchor_class_embeds is not None:
                # if we have an anchor, do it
                combo_embeds = concat_prompt_embeds(
                    [
                        positive_embeds,
                        target_class_embeds,
                        negative_embeds,
                        anchor_embeds,
                    ]
                )
                num_embeds = 4
            else:
                combo_embeds = concat_prompt_embeds(
                    [positive_embeds, target_class_embeds, negative_embeds]
                )
                num_embeds = 3

            # do them in one batch, VRAM should handle it since we are no grad
            combo_pred = self.sd.predict_noise(
                latents=torch.cat([noisy_latents] * num_embeds, dim=0),
                conditional_embeddings=combo_embeds,
                timestep=torch.cat([timesteps] * num_embeds, dim=0),
                guidance_scale=1.0,
                guidance_embedding_scale=1.0,
                batch=batch,
            )

            if self.anchor_class_embeds is not None:
                positive_pred, neutral_pred, negative_pred, anchor_target = (
                    combo_pred.chunk(4, dim=0)
                )
            else:
                anchor_target = None
                positive_pred, neutral_pred, negative_pred = combo_pred.chunk(3, dim=0)

            # calculate the targets
            guidance_scale = self.slider.guidance_strength

            # enhance_positive_target = neutral_pred + guidance_scale * (
            #     positive_pred - negative_pred
            # )
            # enhance_negative_target = neutral_pred + guidance_scale * (
            #     negative_pred - positive_pred
            # )
            # erase_negative_target = neutral_pred - guidance_scale * (
            #     negative_pred - positive_pred
            # )
            # erase_positive_target = neutral_pred - guidance_scale * (
            #     positive_pred - negative_pred
            # )

            positive = (positive_pred - neutral_pred) - (negative_pred - neutral_pred)
            negative = (negative_pred - neutral_pred) - (positive_pred - neutral_pred)

            enhance_positive_target = neutral_pred + guidance_scale * positive
            enhance_negative_target = neutral_pred + guidance_scale * negative
            erase_negative_target = neutral_pred - guidance_scale * negative
            erase_positive_target = neutral_pred - guidance_scale * positive
            
            # normalize to neutral std/mean
            enhance_positive_target = norm_like_tensor(
                enhance_positive_target, neutral_pred
            )
            enhance_negative_target = norm_like_tensor(
                enhance_negative_target, neutral_pred
            )
            erase_negative_target = norm_like_tensor(
                erase_negative_target, neutral_pred
            )
            erase_positive_target = norm_like_tensor(
                erase_positive_target, neutral_pred
            )

            if was_unet_training:
                self.sd.unet.train()

            # restore network
            if self.network is not None:
                self.network.is_active = was_network_active

            if self.anchor_class_embeds is not None:
                # do a grad inference with our target prompt
                embeds = concat_prompt_embeds([target_class_embeds, anchor_embeds]).to(
                    self.device_torch, dtype=dtype
                )

                noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=0).to(
                    self.device_torch, dtype=dtype
                )
                timesteps = torch.cat([timesteps, timesteps], dim=0)
            else:
                embeds = target_class_embeds.to(self.device_torch, dtype=dtype)

        # do positive first
        self.network.set_multiplier(1.0)
        pred = self.sd.predict_noise(
            latents=noisy_latents,
            conditional_embeddings=embeds,
            timestep=timesteps,
            guidance_scale=1.0,
            guidance_embedding_scale=1.0,
            batch=batch,
        )

        if self.anchor_class_embeds is not None:
            class_pred, anchor_pred = pred.chunk(2, dim=0)
        else:
            class_pred = pred
            anchor_pred = None

        # enhance positive loss
        enhance_loss = torch.nn.functional.mse_loss(class_pred, enhance_positive_target)

        erase_loss = torch.nn.functional.mse_loss(class_pred, erase_negative_target)

        if anchor_target is None:
            anchor_loss = torch.zeros_like(erase_loss)
        else:
            anchor_loss = torch.nn.functional.mse_loss(anchor_pred, anchor_target)

        anchor_loss = anchor_loss * self.slider.anchor_strength

        # send backward now because gradient checkpointing needs network polarity intact
        total_pos_loss = (enhance_loss + erase_loss + anchor_loss) / 3.0
        total_pos_loss.backward()
        total_pos_loss = total_pos_loss.detach()

        # now do negative
        self.network.set_multiplier(-1.0)
        pred = self.sd.predict_noise(
            latents=noisy_latents,
            conditional_embeddings=embeds,
            timestep=timesteps,
            guidance_scale=1.0,
            guidance_embedding_scale=1.0,
            batch=batch,
        )

        if self.anchor_class_embeds is not None:
            class_pred, anchor_pred = pred.chunk(2, dim=0)
        else:
            class_pred = pred
            anchor_pred = None

        # enhance negative loss
        enhance_loss = torch.nn.functional.mse_loss(class_pred, enhance_negative_target)
        erase_loss = torch.nn.functional.mse_loss(class_pred, erase_positive_target)

        if anchor_target is None:
            anchor_loss = torch.zeros_like(erase_loss)
        else:
            anchor_loss = torch.nn.functional.mse_loss(anchor_pred, anchor_target)
        anchor_loss = anchor_loss * self.slider.anchor_strength
        total_neg_loss = (enhance_loss + erase_loss + anchor_loss) / 3.0
        total_neg_loss.backward()
        total_neg_loss = total_neg_loss.detach()

        self.network.set_multiplier(1.0)

        total_loss = (total_pos_loss + total_neg_loss) / 2.0

        # add a grad so backward works right
        total_loss.requires_grad_(True)
        return total_loss
