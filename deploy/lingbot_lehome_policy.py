import json
import os
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoConfig
from lerobot.configs.policies import PreTrainedConfig

from lingbotvla.data.vla_data.transform import Normalizer, prepare_images, prepare_language, prepare_state
from lingbotvla.models import build_processor

from .lingbot_robotwin_policy import (
    BASE_MODEL_PATH,
    LingBotVlaInferencePolicy,
    PI0InfernecePolicy,
    load_model_weights,
    merge_qwen_config,
)


class LingbotLehomePolicy:
    def __init__(
        self,
        path_to_pi_model: str,
        task_description: str = "fold the garment on the table",
        use_length: int = 1,
        chunk_ret: bool = False,
        use_bf16: bool = True,
        use_fp32: bool = False,
        training_config_path: str | None = None,
    ) -> None:
        assert not (use_bf16 and use_fp32), "Bfloat16 or Float32!!!"
        self.use_length = use_length
        self.chunk_ret = chunk_ret
        self.use_bf16 = use_bf16
        self.use_fp32 = use_fp32
        self.task_description = task_description
        self.path_to_pi_model = path_to_pi_model
        self.training_config_path = training_config_path

        self.vla = self.load_vla(path_to_pi_model, training_config_path)
        self.vla = self.vla.cuda().eval()
        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
        elif use_fp32:
            self.vla.model.float()

        self.global_step = 0
        self.last_action_chunk = None

    def _resolve_training_config_path(self, path_to_pi_model: str, training_config_path: str | None) -> Path:
        if training_config_path is not None:
            path = Path(training_config_path)
            if not path.exists():
                raise FileNotFoundError(f"Training config not found: {path}")
            return path

        candidates = [
            Path(path_to_pi_model) / "lingbotvla_cli.yaml",
            Path(path_to_pi_model).parent / "lingbotvla_cli.yaml",
            Path(path_to_pi_model).parent.parent / "lingbotvla_cli.yaml",
            Path(path_to_pi_model).parent.parent.parent / "lingbotvla_cli.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Cannot find lingbotvla_cli.yaml near the checkpoint. "
            "Please pass training_config_path explicitly."
        )

    def load_vla(self, path_to_pi_model: str, training_config_path: str | None):
        print(f"loading model from: {path_to_pi_model}")
        config = PreTrainedConfig.from_pretrained(path_to_pi_model)

        resolved_training_config = self._resolve_training_config_path(path_to_pi_model, training_config_path)
        with open(resolved_training_config, "r") as f:
            training_config = yaml.safe_load(f)

        training_model_config = dict(training_config["model"])
        training_model_config.update(training_config["train"])
        for k, v in training_model_config.items():
            setattr(config, k, getattr(config, k, v))

        config.attention_implementation = "eager"

        extra_fields = {
            "resize_imgs_with_padding": [224, 224],
            "adapt_to_pi_aloha": False,
            "use_delta_joint_actions_aloha": False,
            "proj_width": 768,
            "num_steps": 10,
            "use_cache": True,
            "train_state_proj": True,
        }
        for k, v in extra_fields.items():
            if not hasattr(config, k):
                setattr(config, k, v)

        training_base_model = training_config["model"]["tokenizer_path"]
        if "paligemma" in training_base_model:
            model_name = "pi0"
            config.vocab_size = 257152
        elif "qwen2" in training_base_model.lower():
            model_name = "lingbotvla"
        else:
            raise ValueError(f"Unsupported base model of {path_to_pi_model}")

        base_model_path = BASE_MODEL_PATH[model_name]
        config.tokenizer_path = base_model_path
        self.model_name = model_name

        qwen_config = AutoConfig.from_pretrained(base_model_path)
        config = merge_qwen_config(config, qwen_config)

        if "vocab_size" in training_config["model"] and training_config["model"]["vocab_size"] != 0:
            config.vocab_size = training_config["model"]["vocab_size"]

        self.processor = build_processor(base_model_path)
        self.language_tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        data_config = SimpleNamespace(**training_config["data"])

        print("Initializing model ... ")
        if "paligemma" in training_base_model:
            policy = PI0InfernecePolicy(config, tokenizer_path=base_model_path)
        else:
            policy = LingBotVlaInferencePolicy(config, tokenizer_path=base_model_path)

        load_model_weights(policy, path_to_pi_model, strict=True)

        policy.feature_transform = None
        self.data_config = data_config
        self.config = config
        self.joint_max_dim = training_config["train"]["max_action_dim"]
        self.action_dim = training_config["train"]["action_dim"]
        self.chunk_size = training_config["train"]["chunk_size"]
        policy.action_dim = self.action_dim
        policy.chunk_size = self.chunk_size
        self.norm_stats_file = data_config.norm_stats_file
        self.use_depth_align = "align_params" in training_config["train"]

        with open(self.norm_stats_file) as f:
            self.norm_stats = json.load(f)
        policy.normalizer = Normalizer(
            norm_stats=self.norm_stats["norm_stats"],
            from_file=True,
            data_type="customized",
            norm_type={
                "observation.images.top_rgb": "identity",
                "observation.images.left_rgb": "identity",
                "observation.images.right_rgb": "identity",
                "observation.state": self.data_config.norm_type,
                "action": self.data_config.norm_type,
            },
        )

        print("Model initialized ... ")
        return policy

    def reset(self) -> None:
        self.global_step = 0
        self.last_action_chunk = None

        if getattr(self.data_config, "norm_type", None) is None:
            self.data_config.norm_type = "meanstd"
        if getattr(self.config, "vlm_causal", None) is None:
            self.config.vlm_causal = False
        if getattr(self.config, "qwenvl_bos", None) is None:
            self.config.qwenvl_bos = False

    def resize_image(self, observation):
        for image_feature in [
            "observation.images.top_rgb",
            "observation.images.left_rgb",
            "observation.images.right_rgb",
        ]:
            assert image_feature in observation
            image = observation[image_feature]
            assert len(image.shape) == 3
            if image.shape[-1] == 4:
                image = image[:, :, :3]
            assert image.shape[-1] == 3
            img_pil = Image.fromarray(image)
            image_size = getattr(self.data_config, "img_size", 224)
            img_pil = img_pil.resize((image_size, image_size), Image.BILINEAR)
            img_resized = np.transpose(np.array(img_pil), (2, 0, 1))
            observation[image_feature] = img_resized / 255.0

    @torch.no_grad()
    def infer(self, observation):
        self.resize_image(observation)
        for k, v in list(observation.items()):
            if isinstance(v, np.ndarray):
                observation[k] = torch.from_numpy(v)

        if self.use_length == -1 or self.global_step % self.use_length == 0:
            normalized_observation = self.vla.normalizer.normalize(observation)
            top_image = (normalized_observation["observation.images.top_rgb"] * 255).to(torch.uint8)
            left_image = (normalized_observation["observation.images.left_rgb"] * 255).to(torch.uint8)
            right_image = (normalized_observation["observation.images.right_rgb"] * 255).to(torch.uint8)
            obs_dict = {
                "image": {
                    "base_0_rgb": top_image,
                    "left_wrist_0_rgb": left_image,
                    "right_wrist_0_rgb": right_image,
                },
                "state": normalized_observation["observation.state"].to(torch.float32),
                "prompt": [observation.get("task", self.task_description)],
            }
            state = prepare_state(self.config, obs_dict)
            lang_tokens, lang_masks = prepare_language(self.config, self.language_tokenizer, obs_dict)
            images, img_masks, _ = prepare_images(self.config, self.image_processor, obs_dict)
            model_observation = {
                "images": images,
                "img_masks": img_masks,
                "state": state,
                "lang_tokens": lang_tokens,
                "lang_masks": lang_masks,
            }
            if self.use_bf16:
                model_observation["state"] = model_observation["state"].to(torch.bfloat16)
        else:
            model_observation = None

        if self.chunk_ret:
            action = self.vla.select_action(model_observation, self.use_bf16, self.config.vlm_causal)["action"].float().cpu().numpy()
            action = action[: self.use_length, : self.action_dim]
            return action

        if self.use_length == -1 or self.global_step % self.use_length == 0:
            action_chunk = self.vla.select_action(model_observation, self.use_bf16, self.config.vlm_causal)["action"]
            self.last_action_chunk = action_chunk.float().cpu().numpy()

        if self.use_length > 0:
            action = self.last_action_chunk[self.global_step % self.use_length]
        else:
            action = self.last_action_chunk[0]
        action = action[: self.action_dim]
        self.global_step += 1
        return action.astype(np.float32)


def reload_weights(policy, path_to_pi_model: str) -> None:
    all_safetensors = glob(os.path.join(path_to_pi_model, "*.safetensors"))
    merged_weights = {}
    for file_path in tqdm(all_safetensors):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                merged_weights[key] = f.get_tensor(key)
    policy.load_state_dict(merged_weights, strict=True)
