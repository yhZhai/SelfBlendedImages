from collections import OrderedDict
import torch
from src.FlowFormer import build_flowformer
from src.datasets.image_pair_dataset import ImagePairDataset

def main(cfg):
    model = build_flowformer(cfg).cuda()
    state_dict = torch.load("checkpoints/things.pth", map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)

    dataset = ImagePairDataset(
        "train",
        ["data/FaceForensics++/original_sequences/youtube/c23/frames"],
        ["data/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames"],
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
    )

    for i, item in enumerate(train_loader):
        image1 = item["first_frame"].cuda()
        image2 = item["second_frame"].cuda()
        pred_flow = model(image1, image2)
        print(i)
        


if __name__ == "__main__":
    from configs.things import get_cfg
    cfg = get_cfg()
    main(cfg)
