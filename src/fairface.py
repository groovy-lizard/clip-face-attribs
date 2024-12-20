"""FairFace dataset module"""
import pandas as pd


def get_img_list(root_path: str, set_name: str) -> list:
    """Load the given set csv file and return the image list

    :param set: name of the set (train | val)
    :type set: str
    :return: list of image paths
    :rtype: list
    """
    set_df = pd.read_csv(f"{root_path}/fface_{set_name}.csv")
    path_list = set_df['file'].map(str(root_path + "/{}").format)
    return path_list.to_list()
