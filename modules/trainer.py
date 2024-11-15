import os
import torch
import wandb
import evaluate
import numpy as np
import pandas as pd
from peft import LoraConfig
from dotenv import load_dotenv
from omegaconf import OmegaConf
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


class KsatTrainer:
    def __init__(self, model_module, data_module, config):
        self.config = config
        self.data_module = data_module
        self.run_name = f"{config.model_name.replace('/', '-')}_{config.trainer_type}_lr={config.training_params.learning_rate}_bz={config.training_params.batch_size}"
        self.best_accuracy = 0
        self.best_predictions = None
        self.trainer = self._get_trainer(model_module, data_module, config)

    def _get_trainer(self, model_module, data_module, config):
        def preprocess_logits_for_metrics(logits, labels):
            logits = logits if not isinstance(logits, tuple) else logits[0]
            logit_idx = [
                model_module.tokenizer.vocab["1"],
                model_module.tokenizer.vocab["2"],
                model_module.tokenizer.vocab["3"],
                model_module.tokenizer.vocab["4"],
                model_module.tokenizer.vocab["5"],
            ]
            logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
            return logits

        def compute_metrics(evaluation_result):
            # metric 로드
            acc_metric = evaluate.load("accuracy")
            # 정답 토큰 매핑
            int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
            logits, labels = evaluation_result

            # 토큰화된 레이블 디코딩
            labels = np.where(
                labels != -100, labels, model_module.tokenizer.pad_token_id
            )
            labels = model_module.tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )
            labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
            labels = list(map(lambda x: int_output_map[x], labels))

            # calculate predictions
            probs = torch.nn.functional.softmax(torch.tensor(logits).cuda(), dim=-1)
            predictions = np.argmax(probs.cpu(), axis=-1)

            # 정확도 계산
            acc = acc_metric.compute(predictions=predictions, references=labels)

            # best prediction 저장
            if self.best_accuracy < acc["accuracy"]:
                self.best_accuracy = acc["accuracy"]
                self.best_predictions = predictions
            return acc

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=model_module.response_template,
            tokenizer=model_module.tokenizer,
        )

        # 4-2. lora config
        peft_config = LoraConfig(
            r=6,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        load_dotenv()
        checkpoint_dir = os.path.join(os.getenv("ROOT_DIR"), "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        output_dir = os.path.join(checkpoint_dir, self.run_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sft_config = SFTConfig(
            do_train=True,
            do_eval=True,
            lr_scheduler_type="cosine",
            output_dir=output_dir,
            report_to="wandb",
            max_seq_length=config.training_params.max_seq_length,
            per_device_train_batch_size=config.training_params.batch_size,
            per_device_eval_batch_size=config.training_params.batch_size,
            num_train_epochs=config.training_params.num_epochs,
            learning_rate=config.training_params.learning_rate,
            weight_decay=0.01,
            logging_strategy="epoch",
            save_strategy="epoch",
            eval_strategy="epoch",
            save_only_model=True,
            save_total_limit=1,  # 가장 좋은 모델 1개만 유지
            load_best_model_at_end=True,  # 가장 좋은 모델을 학습 종료 시 로드
            metric_for_best_model="accuracy",  # 최고 성능 기준으로 사용할 메트릭
            greater_is_better=True,
            fp16=True,
        )

        trainer = SFTTrainer(
            model=model_module.model,
            train_dataset=data_module.train_examples,
            eval_dataset=data_module.eval_examples,
            tokenizer=model_module.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            args=sft_config,
        )

        return trainer

    def train(self):
        wandb.init(
            project=self.config.wandb.project,
            name=self.run_name,
            entity="seven-eleven",
        )
        self.trainer.train()
        wandb.finish()

        self._save_config()
        self._save_best_eval_predictions()

    def _save_config(self):
        output_dir = os.path.join(os.getenv("ROOT_DIR"), f"checkpoints/{self.run_name}")

        # config.yaml로 저장
        OmegaConf.save(config=self.config, f=os.path.join(output_dir, "config.yaml"))

    def _save_best_eval_predictions(self):
        """
        best acc를 기록한 eval prediction을 predictions 폴더에 저장하는 함수입니다.
        """
        # evaluation에 쓰인 데이터 ids
        ids = [
            self.data_module.eval_dataset["id"][idx]
            for idx in self.data_module.eval_indices
        ]
        eval_prediction_df = pd.DataFrame(
            {
                "id": ids,
                "answer": self.best_predictions,
            }
        )
        eval_prediction_df["answer"] = eval_prediction_df["answer"] + 1

        load_dotenv()
        predictions_dir = os.path.join(os.getenv("ROOT_DIR"), "predictions")
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        output_path = os.path.join(
            predictions_dir,
            f"{self.run_name}_eval_predictions_acc={self.best_accuracy:.04f}.csv",
        )

        eval_prediction_df.to_csv(output_path, index=False)
