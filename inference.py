import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from omegaconf import OmegaConf

from modules.model import KsatModel
from modules.data_module import KsatDataModule


def main():
    load_dotenv()
    # 사용할 checkpoint 폴더명 입력
    run_name = "beomi-gemma-ko-2b_SFT_lr=2e-05_bz=2"
    ROOT_DIR = os.getenv("ROOT_DIR")
    run_path = os.path.join(ROOT_DIR, f"checkpoints/{run_name}")
    sub_dir_list = os.listdir(run_path)
    checkpoint_name = [
        item for item in sub_dir_list if os.path.isdir(os.path.join(run_path, item))
    ][0]
    # /checkpoints/{run_name}/checkpoint-{step}
    checkpoint_path = os.path.join(run_path, checkpoint_name)

    # 1. model, tokenizer 세팅
    model_module = KsatModel(checkpoint_path, use_checkpoint=True)
    model_module.setup()

    # 2. data module 세팅
    config = OmegaConf.load(os.path.join(run_path, "config.yaml"))
    data_module = KsatDataModule(model_module.tokenizer, config.data)
    data_module.setup("test")

    # 3. test dataset prompt데이터로 변환
    test_prompt_dataset = data_module.get_prompt_dataset(data_module.test_dataset)

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
                model_module.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda")
            )

            logits = outputs.logits[:, -1].flatten().cpu()

            target_logit_list = [logits[vocab[str(i + 1)]] for i in range(len_choices)]

            predict_value = pred_choices_map[np.argmax(target_logit_list, axis=-1)]
            infer_results.append({"id": _id, "answer": predict_value})

    # 5. predictions 저장
    output_path = os.path.join(ROOT_DIR, f"predictions/{run_name}_test_predictions.csv")
    pd.DataFrame(infer_results).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
