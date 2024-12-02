import os
import csv
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from omegaconf import OmegaConf
from unsloth import FastLanguageModel

from modules.model import KsatModel
from modules.data_module import KsatDataModule


def main(inference_mode, model_name, use_checkpoint, batch_size):
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
        model_name_or_checkpoint_path, config, use_checkpoint=use_checkpoint
    )
    model_module.setup()
    if not use_checkpoint:
        model_module.model.cuda()

    # 2. data module 세팅
    data_module = KsatDataModule(model_module.tokenizer, config)

    # 3. inference dataset -> prompt 데이터로 변환
    if inference_mode == "test":
        data_module.setup("test")
        test_prompt_dataset = data_module.get_prompt_dataset(data_module.test_dataset)
    else:
        data_module.setup("train")
        test_prompt_dataset = data_module.get_prompt_dataset(
            data_module.eval_dataset.remove_columns(["answer", "solving"])
        )

    # 4. inference
    model_module.tokenizer.padding_side = "left"
    model_module.tokenizer.pad_token_id = model_module.tokenizer.eos_token_id

    # 아웃풋 파일 생성
    df = pd.DataFrame(columns=["id", "solving"])
    output_path = os.path.join(
        ROOT_DIR,
        f"predictions/{model_name.replace('/', '-')}_{inference_mode}_predictions.csv",
    )
    df.to_csv(output_path, index=False, quoting=0)

    # 4-1. with Unsloth
    if config.use_unsloth:
        FastLanguageModel.for_inference(model_module.model)
        for idx in tqdm(range(0, len(test_prompt_dataset), batch_size)):
            batch = test_prompt_dataset[idx : idx + batch_size]
            messages = batch["messages"]
            _id = batch["id"]
            batch_input_ids = model_module.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to("cuda")

            batch_attention_mask = (
                batch_input_ids != model_module.tokenizer.pad_token_id
            ).long()

            outputs = model_module.model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=2048,
                num_beams=1,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                eos_token_id=model_module.tokenizer.eos_token_id,
                pad_token_id=model_module.tokenizer.pad_token_id,
                early_stopping=True,
            )

            for i, output in enumerate(outputs):
                # 입력 프롬프트 이후 새로 생성된 내용만 디코딩
                decoded_output = model_module.tokenizer.decode(
                    output[batch_input_ids[i].shape[0] :], skip_special_tokens=True
                )

                # 출력이 생성될 때마다 csv에 바로 추가
                new_row = [_id[i], decoded_output]
                with open(output_path, mode="a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(new_row)

    # 4-2. withouth Unsloth
    else:
        model_module.model.cuda()
        model_module.model.eval()
        with torch.inference_mode():
            for idx in tqdm(range(0, len(test_prompt_dataset), batch_size)):
                batch = test_prompt_dataset[idx : idx + batch_size]
                messages = batch["messages"]
                _id = batch["id"]
                batch_input_ids = model_module.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to("cuda")

                batch_attention_mask = (
                    batch_input_ids != model_module.tokenizer.pad_token_id
                ).long()

                outputs = model_module.model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
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

                for i, output in enumerate(outputs):
                    # 입력 프롬프트 이후 새로 생성된 내용만 디코딩
                    decoded_output = model_module.tokenizer.decode(
                        output[batch_input_ids[i].shape[0] :], skip_special_tokens=True
                    )

                    # 출력이 생성될 때마다 csv에 바로 추가
                    new_row = [_id[i], decoded_output]
                    with open(
                        output_path, mode="a", newline="", encoding="utf-8"
                    ) as file:
                        writer = csv.writer(file)
                        writer.writerow(new_row)


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
        elif key == "batch_size":
            kwargs[key] = int(value)
        else:
            kwargs[key] = value

    main(
        inference_mode=(
            "test" if "mode" not in kwargs.keys() else kwargs["mode"]
        ),  # validation
        model_name=(
            "unsloth-Qwen2.5-14B-Instruct_CoT_data=default_extract_syn_lr=2e-05_bz=1_acc=0.0000"
            if "model_name" not in kwargs.keys()
            else kwargs["model_name"]
        ),  # 사용할 checkpoint 폴더명 or 사전학습 모델명 입력
        # checkpoint 사용하는 경우 use_checkpoint를 True로, 미학습 모델을 사용하는 경우 False로 설정해주세요.
        use_checkpoint=(
            True if "use_checkpoint" not in kwargs.keys() else kwargs["use_checkpoint"]
        ),
        batch_size=2,  # 적절한 배치 크기로 조정
    )