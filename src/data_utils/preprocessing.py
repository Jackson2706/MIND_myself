# include build-in libraries
import json
import logging
import os

import pandas as pd
from transformers import PreTrainedTokenizer

# include self-defined libraries
from src.data_utils import constants

log = logging.getLogger(__name__)

# Defining the constants values
MAX_SAPO_LENGTH = 64
MAX_TITLE_LENGTH = 32


def category2id(news_path: str = None, dest_path: str = None):
    """
    Create a dictionary to represent the category (row index : 2 in news.tsv) to id
    and save to a json file

    Args:
        news_path: str, path to news.tsv file
        dest_path: str, path to save to json file
    Returns:
        True, dest_path if successful,
        False, None if otherwise
    """
    # if news_path is None
    if news_path is None:
        raise ValueError("news_path must be specified")

    # create destination path if it doesn't exist
    if dest_path is None:
        base_dir = os.path.dirname(news_path)
        filename = "category2id.json"
        dest_path = os.path.join(base_dir, filename)

    try:
        # read the news.tsv file
        news_df = pd.read_csv(news_path, sep="\t", header=None)

        # Get info about the category in the news.tsv file (row index : 2 in news.tsv)
        # and convert it to a list of categories
        elements = list(news_df.get(constants.CATEGORY))
        # Create a dictionary to represent the category
        category2id = {"pad": 0, "unk": 1}

        id_counter = 2

        for element in elements:
            if element not in category2id:
                category2id[element] = id_counter
                id_counter = id_counter + 1

        with open(dest_path, "w") as json_file:
            json.dump(category2id, json_file)

        log.info(
            "Successfully created category2id: {dest_path}".format(
                dest_path=dest_path
            )
        )
        return True, dest_path
    except:
        log.error("Error creating category2id")
        return False, None


def user2id(behavior_path: str = None, dest_path: str = None):
    """
    user ID (row index 1 in behaviors.tsv) is stored via Salt algorithm, datatype: String
    This function is to create a dictionary to represent the user ID (salt algorithm)
    to ID (integer) and save it to a json file

    Args:
        behavior_path: str, path to behavior.tsv file
        dest_path: str, path to save json file
    Returns:
        True, destination path if successful,
        False, None if otherwise
    """
    # if behavior_path is None, then
    if behavior_path is None:
        raise ValueError("behavior_path must be specified")

    # create destination path if it doesn't exist
    if dest_path is None:
        base_dir = os.path.dirname(behavior_path)
        filename = "user2id.json"
        dest_path = os.path.join(base_dir, filename)

    try:
        # Read  behaviors.tsv file
        user_df = pd.read_csv(behavior_path, sep="\t", header=None)

        # Extract user ID from behavior.tsv file, row index 1 in behaviors.tsv
        elements = list(user_df.get(constants.USER_ID))
        user2id = {"unk": 0}

        id_counter = 1

        for element in elements:
            if element not in user2id:
                user2id[element] = id_counter
                id_counter = id_counter + 1
        with open(dest_path, "w") as json_file:
            json.dump(user2id, json_file)

        log.info("Successful created user2id: {}".format(dest_path))
        return True, dest_path
    except:
        log.error("Failed to create user2id")
        return False, None


def encode_sapo(
    news_df: pd.DataFrame = None, tokenizer: PreTrainedTokenizer = None
):
    """
    This function is to encode sapo col information to vector using pre-train language models
    to save memory while training for low-memory PC

    Args:
        news_path: path to the news.tsv file
        tokenizer: a tokenizer to encoder, such as roberta,...

    Returns:
        True, encoded_sapo: List if successful
        False, None if otherwise
    """

    # Check arguments is not empty
    if news_df is None or tokenizer is None:
        raise ValueError("Args must not be specified")
    try:
        # Get the information about sapo column
        sapo_list = list(news_df.get(constants.SAPO))

        # Define a list to store the sapo information sorted like in newss.tsv file
        sapo_encoding_list = []

        for sapo_element in sapo_list:
            sapo_encoding = tokenizer.encode(
                sapo_element,
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_SAPO_LENGTH,
            )
            sapo_encoding_list.append(sapo_encoding)

        logging.info("Sucessfully encoded sapo")
        return True, sapo_encoding_list

    except:
        log.error(
            "Failed to encode sapo col information to vector using pre-train language models"
        )
        return False, None


def encode_titles(
    news_df: pd.DataFrame = None, tokenizer: PreTrainedTokenizer = None
):
    """
    This function is to encode title col information to vector using pre-train language models
    to save memory while training for low-memory PC

    Args:
        news_path: path to the news.tsv file
        tokenizer: a tokenizer to encoder, such as roberta,...

    Returns:
        True, encoded_title: List if successful
        False, None if otherwise
    """

    # Check arguments is not empty
    if news_df is None or tokenizer is None:
        raise ValueError("Args must not be specified")
    try:
        # Get the information about title column
        title_list = list(news_df.get(constants.TITLE))

        # Define a list to store the sapo information sorted like in newss.tsv file
        title_encoding_list = []

        for title_element in title_list:
            title_encoding = tokenizer.encode(
                title_element,
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_TITLE_LENGTH,
            )
            title_encoding_list.append(title_encoding)

        logging.info("Sucessfully encoded title")
        return True, title_encoding_list

    except:
        log.error(
            "Failed to encode title col information to vector using pre-train language models"
        )
        return False, None


def encode_news_data(
    news_path: str = None, tokenizer: PreTrainedTokenizer = None
):
    """
    This function is to encode text format in news.tsv file to vector
    and overwrite this file.
    Args:
        news_path: path to the news.tsv file
        tokenizer: a tokenizer to encoder, such as roberta,...
    Returns:
        True, news_path if successful,
        False, None if otherwise
    """
    if news_path is None or tokenizer is None:
        raise ValueError("Args must not be specified")
    try:
        # Read the news.tsv file
        news_df = pd.read_csv(news_path, sep="\t", header=None)
        sapo_check_flag, encoding_sapo_list = encode_sapo(news_df, tokenizer)
        title_check_flag, encoding_title_list = encode_titles(
            news_df, tokenizer
        )
        if sapo_check_flag:
            news_df[constants.SAPO] = encoding_sapo_list
            log.info("Successfully replaced news.tsv with sapo encoding")
        else:
            log.error("Failed replacing news.tsv with sapo encoding")

        if title_check_flag:
            news_df[constants.TITLE] = encoding_title_list
            log.info("Successfully replaced title.tsv with title encoding")
        else:
            log.error("Failed replacing title.tsv with title encoding")
        news_df.to_csv(news_path, sep="\t", index=False)
        log.info("Successfully preprocessing news.tsv")
        return True, news_path
    except:
        log.error("Failed to preprocess news.tsv")
        return False, None


if __name__ == "__main__":
    # flag, dest_path = category2id()
    # print(dest_path)

    flag, dest_path = user2id("../data/mind-demo/train/behaviors.tsv")
    print(dest_path)

    # encode_news_data(../data/mind-demo/train/behaviors.tsv)
