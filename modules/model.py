import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel


class KsatModel:
    def __init__(self, model_name_or_checkpoint_path, config, use_checkpoint=False):
        self.model_name_or_checkpoint_path = model_name_or_checkpoint_path
        self.config = config
        self.use_checkpoint = use_checkpoint
        self.model = None
        self.tokenizer = None
        self.response_template = None

    def setup(self):
        if not self.config.use_unsloth:
            # setting tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_checkpoint_path,
                trust_remote_code=True,
            )
            if not self.use_checkpoint:
                special_tokens = self._get_special_tokens(
                    self.model_name_or_checkpoint_path
                )
                special_tokens = {"additional_special_tokens": special_tokens}
                self.tokenizer.add_special_tokens(special_tokens)

                if self.tokenizer.chat_template == None:
                    self.tokenizer.chat_template = self._get_chat_template(
                        self.model_name_or_checkpoint_path
                    )

                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.padding_side = "right"

            if self.use_checkpoint:
                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    self.model_name_or_checkpoint_path,
                    trust_remote_code=True,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_checkpoint_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                self.model.resize_token_embeddings(len(self.tokenizer))

            self.response_template = self._get_response_template(
                self.model_name_or_checkpoint_path
            )
        else:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = self.model_name_or_checkpoint_path,
                max_seq_length = self.config.training_params.max_seq_length,
                dtype = torch.float16,
                load_in_4bit = False,
            )

            if not self.use_checkpoint:
                special_tokens = self._get_special_tokens(
                    self.model_name_or_checkpoint_path
                )

                if self.tokenizer.chat_template == None:
                    self.tokenizer.chat_template = self._get_chat_template(
                        self.model_name_or_checkpoint_path
                    )

                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.padding_side = "right"

            self.response_template = self._get_response_template(
                self.model_name_or_checkpoint_path
            )

            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = 16,
                lora_dropout = 0, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )

    @staticmethod
    def _get_special_tokens(model_name):
        if model_name == "beomi/gemma-ko-2b":
            return ["<end_of_turn>", "<start_of_turn>"]
        else:
            return []

    @staticmethod
    def _get_chat_template(model_name):
        # chat template은 반드시 정답 번호<eos token>\n
        if model_name == "beomi/gemma-ko-2b":
            chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>' }}{% endif %}{% endfor %}"
        else:
            chat_template = ""
        return chat_template

    @staticmethod
    def _get_response_template(model_name):
        if model_name == "beomi/gemma-ko-2b":
            response_template = "<start_of_turn>model\n"
        elif model_name == "Bllossom/llama-3.2-Korean-Bllossom-3B":
            response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif model_name == "CarrotAI/Llama-3.2-Rabbit-Ko-3B-Instruct":
            response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif model_name in ["Qwen/Qwen2.5-3B-Instruct", "unsloth/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]:
            response_template = "<|im_start|>assistant\n"
        else:
            response_template = ""
        return response_template
