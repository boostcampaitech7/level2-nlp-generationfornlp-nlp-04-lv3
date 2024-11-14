import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


class KsatModel:
    def __init__(self, model_name_or_checkpoint_path, use_checkpoint=False):
        self.model_name_or_checkpoint_path = model_name_or_checkpoint_path
        self.use_checkpoint = use_checkpoint
        self.model = None
        self.tokenizer = None
        self.response_template = None

    def setup(self):
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
                # torch_dtype=torch.bfloat16,
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

    @staticmethod
    def _get_special_tokens(model_name):
        if model_name == "beomi/gemma-ko-2b":
            return ["<end_of_turn>", "<start_of_turn>"]
        else:
            return []

    @staticmethod
    def _get_chat_template(model_name):
        if model_name == "beomi/gemma-ko-2b":
            chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
        else:
            chat_template = ""
        return chat_template

    @staticmethod
    def _get_response_template(model_name):
        if model_name == "beomi/gemma-ko-2b":
            response_template = "<start_of_turn>model"
        else:
            response_template = ""
        return response_template
