import os
import torch
from torch import nn
from safetensors.torch import load_file as safe_load_file

from model.efficient_net import EffNet3D


class Mapper(nn.Module):
    """
    A simple MLP (Multi-Layer Perceptron) module that maps vision features to llm features.
    """
    def __init__(self, in_channels, out_channels, mlp_depth=1, mlp_bias=True):
        super().__init__()
        modules = nn.ModuleList()
        # Add the first linear layer
        modules.append(nn.Linear(in_channels, out_channels, bias=mlp_bias))
        # Add additional layers if depth > 1
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(out_channels, out_channels, bias=mlp_bias))
        self.layers = nn.Sequential(*modules)

    def forward(self, inp):
        return self.layers(inp)


class CIRCLEReport(nn.Module):
    """
    Main model class that combines visual encoding and language modeling for medical report generation.
    It uses an EfficientNet3D for visual feature extraction and a GPT-based model for text generation.
    """
    def __init__(
        self,
        llm_hidden_size=2048,
        gpt_tokenizer=None,
        gpt_model=None,
        train_gpt=False,
        **kwargs
    ):
        super().__init__()

        # Configuration for the 3D EfficientNet backbone
        cfgs = [
            [1, 32, 4, 1, 0],
            [4, 64, 8, 2, 0],
            [4, 96, 8, 2, 0],
            [4, 192, 16, 2, 1],
            [6, 256, 24, 1, 1],
            [6, 512, 32, 2, 1],
            [6, 640, 8, 1, 1],
        ]
        # Initialize the 3D visual encoder with the specified configuration
        self.visual_encoder = EffNet3D(cfgs, num_classes=37)

        # Mapper to transform visual features to match LLM hidden size
        self.visual_latent_layer = Mapper(1792, llm_hidden_size, mlp_depth=2, mlp_bias=True)

        # Store tokenizer and language model
        self.gpt_tokenizer = gpt_tokenizer
        self.gpt_model = gpt_model
        self.train_gpt = train_gpt  # Flag to determine if GPT should be trained

        # Loss function for training the language model
        self.llm_loss_func = nn.CrossEntropyLoss()

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, vision_path=None, llm_path=None, visual_mapper_path=None, load_clip_weight=False):
        """
        Load pre-trained weights for the visual encoder and/or language model

        Args:
            vision_path: Path to vision encoder weights
            llm_path: Path to language model weights
            visual_mapper_path: Path to visual latent mapper weights
            load_clip_weight: Whether to load CLIP-style weights
        """
        if vision_path is not None:
            assert os.path.exists(vision_path)
            # Load vision encoder weights
            pt = torch.load(vision_path, weights_only=True, map_location="cpu")
            # Handle distributed training weight keys
            if "module." in list(pt.keys())[0]:
                for key in list(pt.keys()):
                    pt[key.replace("module.", "")] = pt.pop(key)
            # Handle CLIP-style weight loading
            if load_clip_weight:
                for key in list(pt.keys()):
                    if 'classifier' in key:
                        pt.pop(key)
                        continue
                    pt[key.replace("visual_transformer.", "")] = pt.pop(key)
            msg = self.visual_encoder.load_state_dict(pt, strict=False)
            print("load vision encoder done, ", msg)
        if visual_mapper_path is not None:
            assert os.path.exists(visual_mapper_path)
            pt = torch.load(visual_mapper_path, weights_only=True, map_location="cpu")
            if "module." in list(pt.keys())[0]:
                for key in list(pt.keys()):
                    pt[key.replace("module.", "")] = pt.pop(key)
            msg = self.visual_latent_layer.load_state_dict(pt, strict=False)
            print("load visual mapper done, ", msg)
        if llm_path is not None:
            # Load LLM weights from safetensors file
            pt = safe_load_file(os.path.join(llm_path, "adapter_model.safetensors"), device="cpu")
            # Handle LoRA weight naming conventions
            for key in list(pt.keys()):
                if "lora_A.weight" in key:
                    pt[key.replace("lora_A.weight", "lora_A.default.weight")] = pt.pop(key)
                if "lora_B.weight" in key:
                    pt[key.replace("lora_B.weight", "lora_B.default.weight")] = pt.pop(key)
            msg = self.gpt_model.load_state_dict(pt, strict=False)
            print("load llm done, ", msg)

    def llm_forward(self, batch_size, device, question_list, answer_list, vision_feature, vision_fea_len):
        """
        Forward pass through the language model component

        Args:
            batch_size: Size of the current batch
            device: Device to run computations on
            question_list: List of questions
            answer_list: List of corresponding answers
            vision_feature: Visual features from the encoder
            vision_fea_len: Length of visual features

        Returns:
            Computed loss for the language model
        """
        text_list = []
        # Combine questions and answers with EOS token
        for i in range(batch_size):
            text_list.append(question_list[i] + answer_list[i] + self.gpt_tokenizer.eos_token)
        # Tokenize the combined text
        text_inputs = self.gpt_tokenizer(text_list, return_tensors="pt", padding=True, max_length=512, truncation=True)
        text_tokens = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # Get text embeddings
        if self.train_gpt:
            text_fea = self.gpt_model.base_model.get_input_embeddings()(text_tokens)
        else:
            text_fea = self.gpt_model.get_input_embeddings()(text_tokens)

        # Concatenate visual and text features
        llm_fea = torch.cat([vision_feature, text_fea], dim=1)
        attention_mask = torch.cat([torch.ones((text_fea.size(0), vision_fea_len)).to(device), attention_mask], dim=1)

        # Forward pass through the GPT model
        llm_outputs = self.gpt_model(inputs_embeds=llm_fea,
                                     attention_mask=attention_mask,
                                     output_hidden_states=False)
        # Extract prediction scores for the answer portion only
        shifted_prediction_scores = llm_outputs.logits[:, vision_fea_len:-1, :].contiguous()
        llm_labels = []
        # Prepare labels by masking out non-answer tokens
        for i in range(batch_size):
            question_input = self.gpt_tokenizer([question_list[i]], return_tensors="pt")
            pre_id = question_input["input_ids"].size(1)
            llm_labels_cur = text_tokens[i]
            end_id = llm_labels_cur.size(0) - 1
            for j in range(llm_labels_cur.size(0) - 1, -1, -1):
                if llm_labels_cur[j] != self.gpt_tokenizer.pad_token_id:
                    end_id = j
                    break
            # Mask out question tokens and padding after answer
            llm_labels_cur[:pre_id] = -100
            llm_labels_cur[end_id + 2:] = -100
            llm_labels.append(llm_labels_cur[1:].contiguous())
        llm_labels = torch.stack(llm_labels, dim=0).contiguous()
        # Compute loss based on whether GPT is being trained
        if self.train_gpt:
            llm_loss = self.llm_loss_func(
                shifted_prediction_scores.view(-1, self.gpt_model.base_model.config.vocab_size), llm_labels.view(-1))
        else:
            llm_loss = self.llm_loss_func(shifted_prediction_scores.view(-1, self.gpt_model.config.vocab_size),
                                          llm_labels.view(-1))
        return llm_loss

    def apply_chat_template_for_no_think(self, prompt):
        """
        Apply chat template without thinking tokens

        Args:
            prompt: Input prompt text

        Returns:
            Formatted text ready for model input
        """
        messages = [{'role': 'user', 'content': prompt}]
        text = self.gpt_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return text

    def report_generation(self, vision_feature, text_list, device):
        """
        Generate medical reports using the model

        Args:
            vision_feature: Visual features from the encoder
            text_list: List of formatted input texts
            device: Device to run computations on

        Returns:
            Generated text responses
        """
        vision_fea_len = vision_feature.size(1)
        # Tokenize input text
        text_inputs = self.gpt_tokenizer(text_list, return_tensors="pt")
        text_tokens = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        # Get text embeddings
        text_fea = self.gpt_model.get_input_embeddings()(text_tokens)
        # Concatenate visual and text features
        llm_fea = torch.cat([vision_feature, text_fea], dim=1)
        pre_len = vision_fea_len
        attention_mask = torch.cat([torch.ones((text_fea.size(0), pre_len)).to(device), attention_mask], dim=1)
        # Generate text using the GPT model
        sample_outputs = self.gpt_model.generate(
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
        # Decode generated tokens to text
        text_outs = []
        for i, sample_output in enumerate(sample_outputs):
            sample_output = sample_output.data.cpu()
            text_outs.append(self.gpt_tokenizer.decode(sample_output, skip_special_tokens=True))
        answer = text_outs[0]
        return answer

    def run_report_generation(self, image, question):
        """
        Run the complete report generation pipeline

        Args:
            image: Input medical image
            question: Question about the image

        Returns:
            Generated medical report
        """
        device = image.device
        # Extract visual features from the image
        vision_feature = self.visual_encoder(image, return_image_feature=True)
        b, c, _, _, _ = vision_feature.size()
        vision_feature = vision_feature.view(b, c, -1)
        vision_feature = vision_feature.permute(0, 2, 1).contiguous()
        vision_feature = self.visual_latent_layer(vision_feature)

        # Format the question using chat template
        text_list = []
        for i in range(vision_feature.size(0)):
            text_list.append(self.apply_chat_template_for_no_think(question))
        # Generate the report
        report_texts = self.report_generation(vision_feature, text_list, device)

        return report_texts

    def forward(
            self,
            image,
            questions,
            answers,
            device,
    ):
        """
        Forward pass for training mode

        Args:
            image: Input medical images
            questions: List of questions about the images
            answers: List of correct answers
            device: Device to run computations on

        Returns:
            Computed loss for training
        """
        # Extract visual features from the image
        vision_feature = self.visual_encoder(image, return_image_feature=True)  # (b, c, d, h, w)
        b, c, _, _, _ = vision_feature.size()
        vision_feature = vision_feature.view(b, c, -1)
        vision_feature = vision_feature.permute(0, 2, 1).contiguous()
        vision_feature = self.visual_latent_layer(vision_feature)

        vision_fea_len = vision_feature.size(1)
        batch_size = vision_feature.size(0)

        # Format questions and prepare answer list
        question_list, answer_list = [], []
        for i in range(batch_size):
            question_list.append(self.apply_chat_template_for_no_think(questions[i]))
            answer_list.append(answers[i])

        # Compute loss using the language model
        llm_loss = self.llm_forward(batch_size, device, question_list, answer_list, vision_feature, vision_fea_len)

        return llm_loss
