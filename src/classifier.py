"""OpenCLIP classifier module"""
import os
from src.encoder import CLIPEncoder
from src import fairface


def run(img_number: int, set_name: str, backbone: str, datasource: str) -> str:
    """Run classification steps by computing logits

    :param img_number: number of image in set
    :type img_number: int
    :param set_name: name of set (train or val)
    :type set_name: str
    :param backbone: model's backbone
    :type backbone: str
    :param datasource: model's datasource
    :type datasource: str
    :return: logits of img and txt embeddings
    :rtype: str
    """
    ff_root = os.environ["FAIRFACE_PATH"]
    imlist = fairface.get_img_list(ff_root, set_name)
    embs_path = f"{ff_root}/embeddings/{set_name}/{backbone}/{datasource}"
    imname = imlist[img_number+1].split("/")[-1].split(".")[0]
    race_dict = fairface.get_race_dict()
    race_list = list(race_dict.keys())
    enc = CLIPEncoder(backbone, datasource)
    img_embs = CLIPEncoder.load_embeddings(f"{embs_path}/{imname}.npy")
    txt_embs = enc.encode_text(race_list)
    return (100.0 * img_embs @ txt_embs.T).softmax(dim=-1)
