import pandas as pd
import os
import json
import logging

log = logging.getLogger(__name__)

def category2id(news_path: str = None, dest_path: str = None):
    '''
        Create a dictionary to represent the category (row index : 2 in news.tsv) to id 
        and save to a json file

        Args:
            news_path: str, path to news.tsv file
            dest_path: str, path to save to json file
        Returns:
            True, dest_path if successful, 
            False, None if otherwise
    '''
    # if news_path is None
    if news_path is None:
        raise ValueError('news_path must be specified')
    
    # create destination path if it doesn't exist
    if dest_path is None:
        base_dir = os.path.dirname(news_path)
        filename = news_path.split('/')[-1].split('.')[0] + ".json"
        dest_path = os.path.join(base_dir, filename)


    try: 
        # read the news.tsv file 
        news_df = pd.read_csv(news_path, sep='\t', header=None)

        # Get info about the category in the news.tsv file (row index : 2 in news.tsv) 
        # and convert it to a list of categories
        elements = list(news_df.get(2))
        # Create a dictionary to represent the category
        category2id = {
            'unk': 0,
            'pad': 1,
        }

        id_counter = 2

        for element in elements:
            if element not in category2id:
                category2id[element] = id_counter
                id_counter = id_counter + 1

        
        with open(dest_path, "w") as json_file:
            json.dump(category2id, json_file)

        log.info("Successfully created category2id: {dest_path}".format(dest_path=dest_path))
        return True, dest_path
    except:
        log.error("Error creating category2id")
        return False, None

def user2id(behavior_path: str = None, dest_path: str = None):
    '''
        user ID (row index 1 in behaviors.tsv) is stored via Salt algorithm, datatype: String
        This function is to create a dictionary to represent the user ID (salt algorithm)
        to ID (integer) and save it to a json file

        Args:
            behavior_path: str, path to behavior.tsv file
            dest_path: str, path to save json file
        Returns:
            True, destination path if successful, 
            False, None if otherwise
    '''
    # if behavior_path is None, then
    if behavior_path is None:
        raise ValueError("behavior_path must be specified")
    
    # create destination path if it doesn't exist
    if dest_path is None:
        base_dir = os.path.dirname(behavior_path)
        filename = behavior_path.split('/')[-1].split('.')[0] + ".json"
        dest_path = os.path.join(base_dir, filename)


    try:
        # Read  behaviors.tsv file
        user_df = pd.read_csv(behavior_path, sep="\t", header=None)
        
        # Extract user ID from behavior.tsv file, row index 1 in behaviors.tsv
        elements = list(user_df.get(1))
        user2id = {
            'unk': 0
        }

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
    
if __name__ == "__main__":
    flag, dest_path = category2id()
    print(dest_path)

    flag, dest_path = user2id("../data/mind-demo/train/behaviors.tsv")