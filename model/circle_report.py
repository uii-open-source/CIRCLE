from collections import OrderedDict
import math
import copy
import os
import random
from contextlib import contextmanager
from functools import partial, wraps
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from safetensors.torch import load_file as safe_load_file
# from contextlib import contextmanager

from model.efficient_net import EffNet3D


class Mapper(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_depth=1, mlp_bias=True):
        super().__init__()
        modules = nn.ModuleList()
        modules.append(nn.Linear(in_channels, out_channels, bias=mlp_bias))
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(out_channels, out_channels, bias=mlp_bias))
        self.layers = nn.Sequential(*modules)

    def forward(self, inp):
        return self.layers(inp)


class CIRCLEReport(nn.Module):
    def __init__(
        self,
        llm_hidden_size=2048,
        gptTokenizer=None,
        gptModel=None,
        train_gpt=False,
        **kwargs
    ):
        super().__init__()

        cfgs = [
            [1, 32, 4, 1, 0],
            [4, 64, 8, 2, 0],
            [4, 96, 8, 2, 0],
            [4, 192, 16, 2, 1],
            [6, 256, 24, 1, 1],
            [6, 512, 32, 2, 1],
            [6, 640, 8, 1, 1],
        ]
        self.visual_encoder = EffNet3D(cfgs, num_classes=37)

        self.visual_latent_layer = Mapper(1792, llm_hidden_size, mlp_depth=2, mlp_bias=True)

        self.gptTokenizer = gptTokenizer
        self.gptModel = gptModel
        self.train_gpt = train_gpt

        self.llm_loss_func = nn.CrossEntropyLoss()

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, vision_path=None, llm_path=None, load_clip_weight=False):
        if vision_path is not None:
            assert os.path.exists(vision_path)
            pt = torch.load(vision_path, weights_only=True, map_location="cpu")
            if "module." in list(pt.keys())[0]:
                for key in list(pt.keys()):
                    pt[key.replace("module.", "")] = pt.pop(key)
            if load_clip_weight:
                for key in list(pt.keys()):
                    if 'classifier' in key:
                        pt.pop(key)
                        continue
                    pt[key.replace("visual_transformer.", "")] = pt.pop(key)
            msg = self.visual_encoder.load_state_dict(pt, strict=False)
            print("load vision encoder done, ", msg)
        if llm_path is not None:
            pt = safe_load_file(os.path.join(llm_path, "adapter_model.safetensors"), device="cpu")
            for key in list(pt.keys()):
                if "lora_A.weight" in key:
                    pt[key.replace("lora_A.weight", "lora_A.default.weight")] = pt.pop(key)
                if "lora_B.weight" in key:
                    pt[key.replace("lora_B.weight", "lora_B.default.weight")] = pt.pop(key)
            msg = self.gptModel.load_state_dict(pt, strict=False)
            print("load llm done, ", msg)

    def llm_forward(self, batchsize, device, question_list, answer_list, vision_feature, vision_fea_len):
        text_list = []
        for i in range(batchsize):
            text_list.append(question_list[i] + answer_list[i] + self.gptTokenizer.eos_token)
        text_inputs = self.gptTokenizer(text_list, return_tensors="pt", padding=True, max_length=512, truncation=True)
        text_tokens = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        if self.train_gpt:
            text_fea = self.gptModel.base_model.get_input_embeddings()(text_tokens)
        else:
            text_fea = self.gptModel.get_input_embeddings()(text_tokens)

        llm_fea = torch.cat([vision_feature, text_fea], dim=1)
        attention_mask = torch.cat([torch.ones((text_fea.size(0), vision_fea_len)).to(device), attention_mask], dim=1)

        llm_outputs = self.gptModel(inputs_embeds=llm_fea,
                                    attention_mask=attention_mask,
                                    output_hidden_states=False)
        shifted_prediction_scores = llm_outputs.logits[:, vision_fea_len:-1, :].contiguous()
        llm_labels = []
        for i in range(batchsize):
            question_input = self.gptTokenizer([question_list[i]], return_tensors="pt")
            pre_id = question_input["input_ids"].size(1)
            llm_labels_cur = text_tokens[i]
            end_id = llm_labels_cur.size(0) - 1
            for j in range(llm_labels_cur.size(0) - 1, -1, -1):
                if llm_labels_cur[j] != self.gptTokenizer.pad_token_id:
                    end_id = j
                    break
            llm_labels_cur[:pre_id] = -100
            llm_labels_cur[end_id + 2:] = -100
            llm_labels.append(llm_labels_cur[1:].contiguous())
        llm_labels = torch.stack(llm_labels, dim=0).contiguous()
        if self.train_gpt:
            llm_loss = self.llm_loss_func(
                shifted_prediction_scores.view(-1, self.gptModel.base_model.config.vocab_size), llm_labels.view(-1))
        else:
            llm_loss = self.llm_loss_func(shifted_prediction_scores.view(-1, self.gptModel.config.vocab_size),
                                          llm_labels.view(-1))
        return llm_loss

    def apply_chat_template_for_no_think(self, prompt):
        messages = [{'role': 'user', 'content': prompt}]
        text = self.gptTokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return text

    def report_generation(self, vision_feature, text_list, device):
        vision_fea_len = vision_feature.size(1)
        text_inputs = self.gptTokenizer(text_list, return_tensors="pt")
        text_tokens = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        text_fea = self.gptModel.get_input_embeddings()(text_tokens)
        llm_fea = torch.cat([vision_feature, text_fea], dim=1)
        pre_len = vision_fea_len
        attention_mask = torch.cat([torch.ones((text_fea.size(0), pre_len)).to(device), attention_mask], dim=1)
        sample_outputs = self.gptModel.generate(
                            inputs_embeds=llm_fea,
                            attention_mask=attention_mask,
                            do_sample=True,
                            temperature=0.2,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=1024,
                            num_return_sequences=1,
                            use_cache=True,
                          )
        text_outs = []
        for i, sample_output in enumerate(sample_outputs):
            sample_output = sample_output.data.cpu()
            text_outs.append(self.gptTokenizer.decode(sample_output, skip_special_tokens=True))
        answer = text_outs[0]
        return answer

    def run_report_generation(self, image, question):
        device = image.device
        vision_feature = self.visual_encoder(image, return_image_feature=True)
        b, c, _, _, _ = vision_feature.size()
        vision_feature = vision_feature.view(b, c, -1)
        vision_feature = vision_feature.permute(0, 2, 1).contiguous()
        vision_feature = self.visual_latent_layer(vision_feature)

        text_list = []
        for i in range(vision_feature.size(0)):
            text_list.append(self.apply_chat_template_for_no_think(question))
        report_texts = self.report_generation(vision_feature, text_list, device)

        return report_texts

    def forward(
            self,
            image,
            questions,
            answers,
            device,
    ):
        vision_feature = self.visual_encoder(image, return_image_feature=True)  # (b, c, d, h, w)
        b, c, _, _, _ = vision_feature.size()
        vision_feature = vision_feature.view(b, c, -1)
        vision_feature = vision_feature.permute(0, 2, 1).contiguous()
        vision_feature = self.visual_latent_layer(vision_feature)

        vision_fea_len = vision_feature.size(1)
        batch_size = vision_feature.size(0)

        question_list, answer_list = [], []
        for i in range(batch_size):
            question_list.append(self.apply_chat_template_for_no_think(questions[i]))
            answer_list.append(answers[i])

        llm_loss = self.llm_forward(batch_size, device, question_list, answer_list, vision_feature, vision_fea_len)

        return llm_loss
