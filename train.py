import os
import hydra
import wandb
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from utils.common import set_seed
from modules.model import KsatModel
from modules.data_module import KsatDataModule
from modules.trainer import KsatTrainer, KsatCoTTrainer, KsatDPOTrainer


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(config):
    set_seed(42)

    if config.trainer_type in ["SFT", "CoT"]:
        # 1. model, tokenizer 세팅
        model_module = KsatModel(config.model_name, config)
        model_module.setup()

        # 2. data module 세팅
        data_module = KsatDataModule(model_module.tokenizer, config)
        data_module.setup("train")

        # 3. trainer 세팅
        if config.trainer_type == "SFT":
            trainer_module = KsatTrainer(model_module, data_module, config)
        elif config.trainer_type == "CoT":
            trainer_module = KsatCoTTrainer(model_module, data_module, config)
    
    elif config.trainer_type == "DPO":
        checkpoint_dir = "Qwen-Qwen2.5-3B-Instruct_CoT_data=default_lr=2e-05_bz=1_acc=0.7013"
        ROOT_DIR = os.getenv("ROOT_DIR")
        run_path = os.path.join(ROOT_DIR, f"checkpoints/{checkpoint_dir}")
        sub_dir_list = os.listdir(run_path)
        checkpoint_name = [
            item for item in sub_dir_list if os.path.isdir(os.path.join(run_path, item))
        ][0]
        # /checkpoints/{run_name}/checkpoint-{step}
        checkpoint_path = os.path.join(run_path, checkpoint_name)

        model_module = KsatModel(checkpoint_path, config)
        model_module.setup()

        data_module = KsatDataModule(model_module.tokenizer, config)
        data_module.setup("train")

        trainer_module = KsatDPOTrainer(model_module, data_module, config)

    # 4. train
    trainer_module.train()


if __name__ == "__main__":
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_KEY"), relogin=True)
    main()
