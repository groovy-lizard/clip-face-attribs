"""Main module entrypoint"""
import os
from dotenv import load_dotenv

from src import encoder
from src import fairface
from src import utils


load_dotenv()
backbone = "ViT-B-32"
source = "openai"
set_name = "train"
enc = encoder.CLIPEncoder(backbone, source)
ff_root = os.environ["FAIRFACE_PATH"]
ff_embs = f"{ff_root}/embeddings/{set_name}"
outfolder = f"{ff_embs}/{backbone}/{source}"
utils.prep_folders(outfolder)
img_list = fairface.get_img_list(ff_root, set_name)
enc.batch_encode_imgs(img_list, outfolder)
