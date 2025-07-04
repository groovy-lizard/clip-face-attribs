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


def get_race_dict() -> dict:
    """Get race dictionary with race names and labels

    :return: race label dictionary
    :rtype: dict
    """
    return {
        "White": 0,
        "Black": 1,
        "Indian": 2,
        "Latino_Hispanic": 3,
        "Southeast Asian": 4,
        "East Asian": 5,
        "Middle Eastern": 6
    }


def get_img_name(file_name):
    """Get image name given file name"""
    return file_name.split("/")[-1].split(".")[0]
