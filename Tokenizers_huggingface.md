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