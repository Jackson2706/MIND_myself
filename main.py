#include built-in libraries
from transformers import AutoTokenizer

# include self-defined libraries
from src.mind import download_extract_small_mind, preprocess

# defining the constants values
PRETRAINED_TOKENIZER = "vinai/phobert-base"

train_path, validation_path = download_extract_small_mind(size="small", 
                                                          dest_path="./data",
                                                          clean_zip_file= False)


print("Training path: {}\nValidation path: {}".format(train_path, validation_path))

Tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER)
flag = preprocess(train_path=train_path, tokenizer=Tokenizer)
print(flag)