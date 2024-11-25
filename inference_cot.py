import os
import torch
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from omegaconf import OmegaConf
import argparse

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
        data_module.setup("train")
        test_prompt_dataset = data_module.get_prompt_dataset(
            data_module.eval_dataset.remove_columns(["answer", "solving"])
        )

    # 4. inference
    model_module.model.cuda()
    model_module.model.eval()
    solvings = []
    ids = []
    with torch.inference_mode():
        for data in tqdm(test_prompt_dataset):
            _id = data["id"]
            messages = data["messages"]

            input_ids = model_module.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")

            attention_mask = (input_ids != model_module.tokenizer.pad_token_id).long()

            outputs = model_module.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                num_beams=5,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                eos_token_id=model_module.tokenizer.eos_token_id,
                pad_token_id=model_module.tokenizer.pad_token_id,
                early_stopping=True,
            )

            # 입력 프롬프트 이후 새로 생성된 내용만 디코딩
            decoded_output = model_module.tokenizer.decode(
                outputs[0][input_ids.shape[1] :], skip_special_tokens=True
            )

            solvings.append(decoded_output)
            ids.append(_id)

    df = pd.DataFrame(
        {
            "id": ids,
            "solving": solvings,
        }
    )

    # 5. predictions 저장
    output_path = os.path.join(
        ROOT_DIR,
        f"predictions/{model_name.replace('/', '-')}_{inference_mode}_predictions.csv",
    )
    df.to_csv(output_path, index=False, quoting=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="*")
    args = parser.parse_args()
    kwargs = {}
    for pair in args.pairs:
        key, value = pair.split("=")
        if key == "use_checkpoint" and value == "False":
            kwargs[key] = False
        elif key == "use_checkpoint" and value == "True":
            kwargs[key] = True
        else:
            kwargs[key] = value

    main(
        inference_mode=(
            "test" if "mode" not in kwargs.keys() else kwargs["mode"]
        ),  # validation
        model_name=(
            "Bllossom-llama-3.2-Korean-Bllossom-3B_SFT_data=cot_lr=2e-05_bz=1_acc=0.9115"
            if "model_name" not in kwargs.keys()
            else kwargs["model_name"]
        ),  # 사용할 checkpoint 폴더명 or 사전학습 모델명 입력
        # checkpoint 사용하는 경우 use_checkpoint를 True로, 미학습 모델을 사용하는 경우 False로 설정해주세요.
        use_checkpoint=(
            True if "use_checkpoint" not in kwargs.keys() else kwargs["use_checkpoint"]
        ),
    )
