import hydra
from omegaconf import DictConfig
import segmentation


@hydra.main(version_base=None, config_path="configs", config_name="medseg.yaml")
def main(cfg: DictConfig):
    if cfg.stage == "train":
        segmentator = segmentation.__dict__[f'{cfg.segmentator}'](cfg, is_train=True)
        segmentator.train()
    elif cfg.stage == "test":
        segmentator = segmentation.__dict__[f'{cfg.segmentator}'](cfg, is_train=False)
        segmentator.test()
    elif cfg.stage == "inference":
        segmentator = segmentation.__dict__[f'{cfg.segmentator}'](cfg, is_train=False)
        segmentator.inference()
    else:
        raise ValueError("Wrong stage!")

if __name__ == "__main__":
    main()
