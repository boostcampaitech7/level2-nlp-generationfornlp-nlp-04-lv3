import os
import hydra
import wandb
from dotenv import load_dotenv

from utils.common import set_seed
from modules.model import KsatModel
from modules.trainer import KsatTrainer
from modules.data_module import KsatDataModule


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(config):
    set_seed(42)

    # 1. model, tokenizer 세팅
    model_module = KsatModel(config.model_name)
    model_module.setup()

    # 2. data module 세팅
    data_module = KsatDataModule(model_module.tokenizer, config.data)
    data_module.setup("train")

    # 3. trainer 세팅
    trainer_module = KsatTrainer(model_module, data_module, config)

    # 4. train
    trainer_module.train()


if __name__ == "__main__":
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_KEY"), relogin=True)
    main()
