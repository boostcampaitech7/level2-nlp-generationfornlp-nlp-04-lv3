import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from omegaconf import OmegaConf

from modules.model import KsatModel
from modules.data_module import KsatDataModule


def main(inference_mode, model_name, use_checkpoint):
    load_dotenv()

    # 0. 모델 경로 설정
    ROOT_DIR = os.getenv("ROOT_DIR")
    if use_checkpoint:
        run_path = os.path.join(ROOT_DIR, f"checkpoints/{model_name}")
        sub_dir_list = os.listdir(run_path)
        checkpoint_name = [
            item for item in sub_dir_list if os.path.isdir(os.path.join(run_path, item))
        ][0]
        # /checkpoints/{run_name}/checkpoint-{step}
        checkpoint_path = os.path.join(run_path, checkpoint_name)
        model_name_or_checkpoint_path = checkpoint_path

        config = OmegaConf.load(os.path.join(run_path, "config.yaml"))
    else:
        model_name_or_checkpoint_path = model_name
        config = OmegaConf.load(os.path.join(ROOT_DIR, "config/config.yaml"))

    # 1. model, tokenizer 세팅
    model_module = KsatModel(
        model_name_or_checkpoint_path, use_checkpoint=use_checkpoint
    )
    model_module.setup()
    if not use_checkpoint:
        model_module.model.cuda()

    # 2. data module 세팅
    data_module = KsatDataModule(model_module.tokenizer, config.data)

    # inference dataset -> prompt 데이터로 변환
    if inference_mode == "test":
        data_module.setup("test")
        test_prompt_dataset = data_module.get_prompt_dataset(data_module.test_dataset)
    else:
        data_module.setup("validation")
        test_prompt_dataset = data_module.get_prompt_dataset(
            data_module.eval_dataset.remove_columns(["answer"])
        )

    # 4. inference
    infer_results = []
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    vocab = model_module.tokenizer.vocab
    model_module.model.eval()
    with torch.inference_mode():
        for data in tqdm(test_prompt_dataset):
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]

            outputs = model_module.model(
                input_ids=model_module.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda"),
            )

            logits = outputs.logits[:, -1].flatten().cpu()

            target_logit_list = [logits[vocab[str(i + 1)]] for i in range(len_choices)]

            predict_value = pred_choices_map[np.argmax(target_logit_list, axis=-1)]
            infer_results.append({"id": _id, "answer": predict_value})

    # 5. predictions 저장
    output_path = os.path.join(
        ROOT_DIR, f"predictions/{model_name}_{inference_mode}_predictions.csv"
    )
    pd.DataFrame(infer_results).to_csv(output_path, index=False)


if __name__ == "__main__":
    main(
        inference_mode="test",  # validation
        model_name="Bllossom/llama-3.2-Korean-Bllossom-3B",  # 사용할 checkpoint 폴더명 or 사전학습 모델명 입력
        # checkpoint 사용하는 경우 use_checkpoint를 True로, 미학습 모델을 사용하는 경우 False로 설정해주세요.
        use_checkpoint=False,
    )
