# Hugginface - Tokenizers
- the Rust implemetation 덕에 굉장히 빠른 훈련과 토크나이즈가 가능
- Full alignment tracking: 파괴적인 정규화를 하더라도 원래 문장의 일부분을 항상 되찾을 수 있음
- truancation, padding, add the special tokens 등 전처리 가능

## Quick tour

BPE모델 인스턴스화
```
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
```

학습을 위해서는 trainer 인스턴스를 써야함
inputs = [vocab_size, min_frequency, special_tokens]
```
from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
```
- special tokens의 순서 매우중요
  - 순서에 따라 ID가 달라지기 때문
  - 현재 순서는 [UNK]: 0, [CLS]: 1

사전 토큰화기를 사용하면 사전 토큰화기에서 나오는 단어보다 큰 토큰은 안나오게 됨

```
# 공백으로 나누는 가장 간단한 pre-tokenizer
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()
```

train 메소드에 file과 trainer를 입력

```
files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
tokenizer.train(files, trainer)
```

tokenizer의 configuration과 vocab 저장
```
tokenizer.save("data/tokenizer-wiki.json")
```

tokenizer 불러오기
```
tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")
```

encode 메소드로 토크나이저 사용 가능
```
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
```
=> 'Encoding' object 리턴
tokens, ids 속성
```
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
print(output.ids)
# [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]
```
offset으로 [UNK] 토큰 원래 값 찾기 가능 
```
print(output.offsets[9])
# (26, 27)

sentence = "Hello, y'all! How are you 😁 ?"
sentence[26:27]
# "😁"
```

Post-processing: [CLS], [SEP] 후 삽입
TemplateProcessing이 가장 흔하게 사용

우선 더블체크를 위해 토큰 id 먼저 확인
```
tokenizer.token_to_id("[SEP]")
# 2
```
traditional BERT를 주는 post-processing
```
from tokenizers.processors import TemplateProcessing

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)
```
- '\$A'가 sentence를 나타냄. $B는 second sentence
- 뒤에오는 ':1'은 sentence ID. default는 0
- 스페셜 토큰을 vocab과 연결

results
```
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"]

output = tokenizer.encode("Hello, y'all!", "How are you 😁 ?")
print(output.tokens)
# ["[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]", "How", "are", "you", "[UNK]", "?", "[SEP]"]

print(output.type_ids)
# [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
```
- output.type_ids 는 세그먼트 정보

최고의 속도를 내기위해선 'encode_batch()' 메소드로 batch를 사용하는 것이 좋음
```
output = tokenizer.encode_batch(
    [["Hello, y'all!", "How are you 😁 ?"], ["Hello to you too!", "I'm fine, thank you!"]]
)
```
- 리스트로 입력

enable_padding 메소드로 패드토큰만 입력하면, 배치 중 가장 긴 문장 기준으로 패딩 입력
```
tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
```
 - direction: 패딩 방향 설정. 디폴트는 오른쪽
 - length: 모든 문장 길이 고정하고 싶을때 지정
  
```
output = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
print(output[1].tokens)
# ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]

print(output[1].attention_mask)
# [1, 1, 1, 1, 1, 1, 1, 0]
```
- attention_mask: 문장과 패딩 구분

프리트레인드 토크나이저
```
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
```

직접 vocab 파일을 다운받아서 사용도 가능
```
!wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
```