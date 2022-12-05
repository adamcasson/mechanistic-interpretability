import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path='mechanistic_interpretability/configs',
    config_name='main',
)
def main(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.task.datamodule)
    model = instantiate(cfg.task.model)
    trainer = instantiate(cfg.task.trainer)

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()
