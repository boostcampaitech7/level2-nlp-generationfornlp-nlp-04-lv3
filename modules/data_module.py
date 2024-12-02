import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from ast import literal_eval
from dotenv import load_dotenv

from utils import generate_prompt


class KsatDataModule:
    def __init__(self, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer

        # raw dataset(id, paragraph, question, choices, answer, question_plus)
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

        # raw dataset => convert to prompt (id, message, label) => tokenized dataset
        self.train_examples = None
        self.train_indices = None
        self.eval_examples = None
        self.eval_indices = None
        self.test_examples = None
        self.test_indices = None

    def setup(self, mode="train"):
        if mode == "train":
            self.train_dataset = self.get_dataset("train")

            if self.config.trainer_type == "DPO":
                self.train_examples, self.train_indices = self.get_dpo_examples(
                    self.train_dataset
                )
            else:
                self.eval_dataset = self.get_dataset("validation")   
                self.train_examples, self.train_indices = self.get_examples(
                    self.train_dataset
                )
                self.eval_examples, self.eval_indices = self.get_examples(self.eval_dataset)
        else:
            self.test_dataset = self.get_dataset("test")

            if self.config.trainer_type == "DPO":
                self.test_examples, self.test_indices = self.get_dpo_examples(self.test_dataset)
            else:
                self.test_examples, self.test_indices = self.get_examples(self.test_dataset)

    # 원본 데이터셋
    def get_dataset(self, split="train"):
        load_dotenv()
        ROOT_DIR = os.getenv("ROOT_DIR")
        if split == "test":
            raw_df = pd.read_csv(f"{ROOT_DIR}/data/test.csv")
        else:
            raw_df = pd.read_csv(
                f"{ROOT_DIR}/data/{self.config.data.dataset_name}/{split}.csv"
            )

        records = []
        for i, row in raw_df.iterrows():
            problems = literal_eval(row["problems"])
            record = {
                "id": row["id"],
                "paragraph": row["paragraph"],
                "question": problems["question"],
                "choices": problems["choices"],
                "answer": problems.get("answer", None),
                "question_plus": problems.get("question_plus", None),
            }
            if "question_plus" in problems:
                record["question_plus"] = problems["question_plus"]
    
            if split != "test" and self.config.trainer_type == "CoT":
                record["solving"] = row["claude-solving"]
            elif self.config.trainer_type == "DPO":
                record["chosen"] = row["claude-solving"]
                record["rejected"] = row["openai-solving"] if i%2 else row["qwen-solving"]
            records.append(record)

        # Convert to DataFrame
        formatted_df = pd.DataFrame(records)
        dataset = Dataset.from_pandas(formatted_df)
        return dataset

    # columns: input_ids, attention_mask
    def get_examples(self, dataset):
        prompt_dataset = self.get_prompt_dataset(dataset)
        tokenized_dataset = prompt_dataset.map(
            self.tokenize,
            remove_columns=list(prompt_dataset.features),
            batched=True,
            batch_size=self.config.data.batch_size,
            num_proc=self.config.data.preprocessing_num_workers,
            load_from_cache_file=True,
            desc="Tokenizing",
        )

        filtered_dataset = []
        filtered_indices = []
        for idx, example in enumerate(tokenized_dataset):
            if len(example["input_ids"]) <= self.config.data.max_seq_length:
                filtered_dataset.append(example)
                filtered_indices.append(idx)
        filtered_dataset = Dataset.from_list(filtered_dataset)
        return filtered_dataset, filtered_indices
    
    def get_dpo_examples(self, dataset):
        prompt_dataset = self.get_prompt_dataset(dataset)
        tokenized_dataset = prompt_dataset.map(
            self.tokenize_dpo,
            remove_columns=list(prompt_dataset.features),
            batched=True,
            batch_size=self.config.data.batch_size,
            num_proc=self.config.data.preprocessing_num_workers,
            load_from_cache_file=True,
            desc="Tokenizing",
        )

        filtered_dataset = []
        filtered_indices = []
        for idx, example in enumerate(tokenized_dataset):
            if len(example["prompt_input_ids"] + example["chosen_input_ids"]) <= self.config.data.max_seq_length or len(example["prompt_input_ids"] + example["rejected_input_ids"]) <= self.config.data.max_seq_length:
                filtered_dataset.append(example)
                filtered_indices.append(idx)
        filtered_dataset = Dataset.from_list(filtered_dataset)
        return filtered_dataset, filtered_indices

    # columns: id, message, label, len_choice
    def get_prompt_dataset(self, dataset):
        # Tokenization and other preprocessing
        prompt_dataset = []
        for data in tqdm(dataset, desc="Converting to prompts"):
            prompt_dataset.append(
                getattr(generate_prompt, self.config.data.prompt_func)(data)
            )

        prompt_dataset = Dataset.from_pandas(pd.DataFrame(prompt_dataset))

        return prompt_dataset

    def _formatting_prompts_func(self, data, key="messages"):
        output_texts = []
        for i in range(len(data[key])):
            output_text = self.tokenizer.apply_chat_template(
                    data[key][i],
                    tokenize=False,
                )
            if key != "prompt":
                output_text = output_text.strip()

            output_texts.append(output_text)
        return output_texts

    def tokenize(self, data, key="messages"):
        outputs = self.tokenizer(
            self._formatting_prompts_func(data, key),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
            max_length=self.config.data.max_seq_length,
            stride=self.config.data.doc_stride,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    def tokenize_dpo(self, data):
        tokenized_promt = self.tokenize(data, "prompt")
        tokenized_chosen = self.tokenize(data, "chosen")
        tokenized_rejected = self.tokenize(data, "rejected")

        return {
            "prompt_input_ids": tokenized_promt["input_ids"],
            "prompt_attention_mask": tokenized_promt["attention_mask"],
            "chosen_input_ids": tokenized_chosen["input_ids"],
            "chosen_attention_mask": tokenized_chosen["attention_mask"],
            "rejected_input_ids": tokenized_rejected["input_ids"],
            "rejected_attention_mask": tokenized_rejected["attention_mask"],
        }
