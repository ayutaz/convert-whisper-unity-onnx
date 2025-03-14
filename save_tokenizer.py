from transformers import WhisperTokenizer

# 多言語用 tiny モデルのトークナイザ
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

# tokenizer.json / merges.txt / vocab.json などが保存される
tokenizer.save_pretrained("./multilingual_tokenizer")
