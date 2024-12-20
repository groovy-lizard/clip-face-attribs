"""openCLIP encoder module"""
import torch
from tqdm import tqdm
from PIL import Image

from src import utils


class CLIPEncoder():
    """openCLIP encoder class"""

    def __init__(self, backbone, datasource):
        model_dict = utils.load_open_clip(backbone, datasource)
        self.model = model_dict['Model']
        self.preprocessing = model_dict['Preprocessing']
        self.device = model_dict['Device']
        self.tokenizer = model_dict['Tokenizer']

    def encode_text(self, txt_list: list):
        """Encode text embeddings based on a text list

        :param txt_list: a list of text labels to be encoded
        :type txt_list: list[str]
        :return: encoded text features
        :rtype: torch.tensor
        """
        text_inputs = torch.cat(
            [self.tokenizer(c) for c in txt_list]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def encode_image(self, impath: str):
        """Encode image

        :param impath: image path
        :type impath: str
        :return: encoded image features
        :rtype: torch.tensor
        """
        image = Image.open(impath)
        img_input = self.preprocessing(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(img_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features

    def batch_encode_imgs(self, imlist: list, outfolder: str):
        """Encode a list of images and save it to disc

        :param imlist: list of image paths
        :type imlist: list
        :param outfolder: embedding's output folder
        :type outfolder: str
        """
        print("Encoding images...")
        for impath in tqdm(imlist):
            imname = impath.split("/")[-1].split(".")[0]
            image_features = self.encode_image(impath)
            torch.save(image_features, f"{outfolder}/{imname}.npy")

    def load_embeddings(self, emb_path):
        """Load embeddings given path"""
        return torch.load(emb_path)
