"""Main module entrypoint"""
import os
import argparse
from dotenv import load_dotenv

from src import encoder
from src import fairface
from src import utils


load_dotenv()


def run_encoder(model, source, set_name):
    """Run the encoder to generate img embeddings"""
    print("Running encoder...")
    ff_root = os.environ["FAIRFACE_PATH"]
    ff_embs = f"{ff_root}/embeddings/{set_name}"
    outfolder = f"{ff_embs}/{model}/{source}"
    utils.prep_folders(outfolder)
    img_list = fairface.get_img_list(ff_root, set_name)
    enc = encoder.CLIPEncoder(model, source)
    enc.batch_encode_imgs(img_list, outfolder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="clip_face_attribs",
        description="Classify facial attributes using CLIP",
        epilog="Questions e-mail me at lmceschini@inf.ufrgs.br"
    )

    parser.add_argument("-b", "--backbone", required=True)
    parser.add_argument("-d", "--datasource", required=True)
    parser.add_argument("-s", "--split", required=True,
                        choices=["train", "val"])
    parser.add_argument("-l", "--labels", required=True,
                        choices=["raw_gender", "raw_race", "original_clip",
                                 "ARGP"])
    parser.add_argument("-se", "--skip_encode", action="store_true")

    args = parser.parse_args()
    backbone, datasource, split = args.backbone, args.datasource, args.split
    if not args.skip_encode:
        run_encoder(backbone, datasource, split)
    labels_path = args.labels
