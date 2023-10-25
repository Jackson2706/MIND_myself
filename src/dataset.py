from typing import List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from collections import OrderedDict

class News:
    '''
        Create a object representing news data
    '''
    #TODO: Understanding the arguments in the constructor -> done
    def __init__(self, news_id: str, title: List[int], sapo: List[int], category: int):
        self._news_id = news_id # Row 0 in news.tsv file
        self._title = title # Row 1 in news.tsv file
        self._sapo = sapo # Row 3 in news.tsv file
        self._category = category #Row 2 in news.tsv file

    @property
    def news_id(self):
        return self.news_id
    
    @property
    def title(self):
        return self._title
    
    @property
    def sapo(self):
        return self._sapo
    
    @property
    def category(self):
        return self._category
    

class Impression:
    '''
        Create a object representing impression log 
    '''

    def __init__(self, impression_id: int, user_id: int, news: List[News], label: List[int]):
        self._impression_id = impression_id
        self._user_id = user_id
        self._news = news
        self._label = label

        @property
        def impression_id(self):
            return self._impression_id
        
        @property
        def user_id(self):
            return self._user_id
        
        @property
        def news(self):
            return self._news
        
        @property
        def label(self):
            return self._label
        
class Behavior:
    '''
         Create a object representing behavior data
    '''
    def __init__(self, behavior_id: int, user_id: int, clicked_news: List[News], impression: List[Impression]):
        self._beahvior_id = behavior_id
        self._user_id = user_id
        self._clicked_news = clicked_news
        self._impression = impression

    @property
    def behavior_id(self):
        return self._beahvior_id
    
    @property
    def user_id(self):
        return self._user_id
    
    @property
    def clicked_news(self):
        return self._clicked_news
    
    @property
    def impression(self):
        return self._impression


class MyDataset(Dataset):
    '''
        Create a dataset to prepare for training
    '''
    
    # Create mode of dataset
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, data_name: str, tokenizer: PreTrainedTokenizer, category2id: dict):
        super.__init__()
        self._name = data_name
        self._samples = OrderedDict()
        self._mode = MyDataset.TRAIN_MODE
        self._tokenizer = tokenizer
        self._category2id = category2id

        self._news_id = 0
        self._id = 0

    def set_mode(self, mode: str):
        self._mode = mode

    def create_news(self):, title: List[str], sapo
