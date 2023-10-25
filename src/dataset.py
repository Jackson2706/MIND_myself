from typing import List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from collections import OrderedDict



from src import utils

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

    def create_news(self, tittle: List[int], sapo: List[int], category: int):
        news = News(news_id=self._news_id, tittle=tittle, sapo=sapo, category=category)
        self._news_id += 1

        return news
    
    @staticmethod    # Call this method without arguments self
    def create_impression(impression_id: int, user_id: int, news: List[News], label: List[int]):
        impression = Impression(impression_id=impression_id,
                                user_id=user_id,
                                news=news,
                                label=label)
        return impression
    

    def add_sample(self, user_id: int, clicked_news: List[News], impression: Impression):
        sample = Behavior(behavior_id=self._id, 
                          user_id=user_id,
                          clicked_news=clicked_news,
                          impression=impression)
        self._samples[self._id] = sample
        self._id += 1
    

    @property
    def samples(self):
        return list(self._samples.values())
    
    @property
    def new_count(self):
        return self._news_id
    
    def __len__(self):
        return len(self._samples)
    
    def __getitem__(self, index: int):
        sample = self.samples[index]

        #TODO: Create samples list for train and val phase

    

    def _create_sample(sample: Behavior, tokenizer: PreTrainedTokenizer, category2id: dict):
    title_clicked_news_encoding = [news.title for news in sample.clicked_news]
    sapo_clicked_news_encoding = [news.sapo for news in sample.clicked_news]
    category_clicked_news_encoding = [news.category for news in sample.clicked_news]
    title_impression_encoding = [news.title for news in sample.impression.news]
    sapo_impression_encoding = [news.sapo for news in sample.impression.news]
    category_impression_encoding = [news.category for news in sample.impression.news]
        
     # Create tensor
    impression_id = torch.tensor(sample.impression.impression_id)
    title_clicked_news_encoding = utils.padded_stack(title_clicked_news_encoding, padding=tokenizer.pad_token_id)
    sapo_clicked_news_encoding = utils.padded_stack(sapo_clicked_news_encoding, padding=tokenizer.pad_token_id)
    category_clicked_news_encoding = torch.tensor(category_clicked_news_encoding)
    his_mask = (category_clicked_news_encoding != category2id['pad'])
    his_title_mask = (title_clicked_news_encoding != tokenizer.pad_token_id)
    his_sapo_mask = (sapo_clicked_news_encoding != tokenizer.pad_token_id)

    title_impression_encoding = utils.padded_stack(title_impression_encoding, padding=tokenizer.pad_token_id)
    sapo_impression_encoding = utils.padded_stack(sapo_impression_encoding, padding=tokenizer.pad_token_id)
    category_impression_encoding = torch.tensor(category_impression_encoding)
    title_mask = (title_impression_encoding != tokenizer.pad_token_id)
    sapo_mask = (sapo_impression_encoding != tokenizer.pad_token_id)

    label = torch.tensor(sample.impression.label)

    return dict(his_title=title_clicked_news_encoding, his_title_mask=his_title_mask,
                his_sapo=sapo_clicked_news_encoding, his_sapo_mask=his_sapo_mask,
                his_category=category_clicked_news_encoding, his_mask=his_mask, title=title_impression_encoding,
                title_mask=title_mask, sapo=sapo_impression_encoding, sapo_mask=sapo_mask,
                category=category_impression_encoding, impression_id=impression_id, label=label)


def create_train_sample(sample: Sample, tokenizer: PreTrainedTokenizer, num_category: int):
    return _create_sample(sample, tokenizer, num_category)


def create_eval_sample(sample: Sample, tokenizer: PreTrainedTokenizer, num_category: int):
    return _create_sample(sample, tokenizer, num_category)
