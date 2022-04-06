# 1. 처음 만나는 자연어 처리

## 1-1. 딥러닝 기반 자연어 처리 모델

## 1-2. 트랜스퍼 러닝

업스트림 태스크
- 다음 단어 맞히기: GPT 계열 업스트림 태스크, 해당 태스크로 학습된 모델을 "언어모델(language model)"이라고 함
- 빈칸 채우기: BERT 계열 업스트림 태스크, 해당 태스크로 학습된 모델을 "마스크 언어모델(masked language model)"이라고 함
  - 위 두가지 태스크는 자기지도학습(self-supervised learning)임

다운스트림 태스크
- 문서분류
- 자연어추론: 두 문장 사이의 관계 추론(참, 거짓, 중립 등)
- 개체명인식
- 질의응답
- 문장생성
  - 위 태스크들은 모두 파인튜닝 방식으로 학습됨
  - 그러나 실제로 다양한 다운스트림 태스크 학습 방식 존재
    - 파인튜닝(fine-tuning): 다운스트림 태스크 데이터 전체 사용. 데이터에 맞게 모델 전체 업데이트. 비용문제 있음
    - 프롬프트 튜닝(prompt tuning): 다운스트림 태스크 데이터 전체 사용. 데이터에 맞게 모델 일부만 업데이트
    - 인컨텍스트 러닝(in-context learning): 다운스트림 태스크 데이터 일부 사용. 모델을 업데이트 하지 않음
      - 제로샷 러닝(zero-shot learning): 데이터 0건 사용
      - 원샷 러닝(one-shot learning): 데이터 1건 사용
      - 퓨샷 러닝(few-shot learning): 데이터 몇건 사용


## 1-3. 학습 파이프라인 소개
1. 각종 설정값 정하기
   - args(lr, batch size, ...)
2. 데이터 내려받기
   - Korpora 패키지는 다양한 한국어 말뭉치 쉽게 내려받고 전처리하도록 도와줌
3. 프리트레인을 마친 모델 준비하기
4. 토크나이저 준비하기
5. 데이터 로더 준비하기
   - 동일한 배치 안에 있는 토큰만 길이가 같으면 됨
   - 컬레이트(collate): 배치의 모양 등을 정비해 모델의 최종 입력으롬나들어 주는 과정
     - list -> tensor 변환도  포함
6. 태스크 정의하기
   - 태스크 = (모델, 최적화방법, 학습과정 ...)
7. 모델 학습하기
   - trainer: 파이토치 라이트닝에서  제공하는 객체로 실제 학습을 수행
     - GPU 등 하드웨어 설정, 학습 기록 로깅, 체크포인트 저장 등 복잡한 설정을 알아서 해줌

## 1-4. 개발 환경 설정
- 구글 드라이브와 연결
  ```
  from google.colab import drive
  drive.mount('/gdrive', force_remount=True)
  ```


# 2. 문장을 작은 단위로 쪼개기

## 2-1. 토큰화란?
- 토큰화: 문장 -> 토큰 시퀀스
- 토큰화 개념을 넓은 의미로 해석할 때, 토큰 나누기 + 품사 부착
- 토큰화 방식
  1. 단어 단위 토큰화
     - 어휘 집합의 크기가 매우 커질 수 있음
  2. 문자 단위 토큰화
     - 어휘 집합의 크기 문제로부터 상대적으로 자유로움
     - 미등록 토큰 문제로부터 자유로움
     - BUT 각 문자 토큰은 의미 있는 단위가 되기 어려움
     - 또한 토큰 시퀀스의 길이가 상대적으로 김 -> 성능 악화
  3. 서브워드 단위 토큰화
     - 단어와 문자 단위 토큰화의 중간에 있는 형태 -> 둘의 장점만을 취함
     - 어휘 집합 크기가 지나치게 커지지 않으면서도 미등록 토큰 문제를 해결, 분석된 토큰 시퀀스가 너무 길어지지 않게 함
     - 대표적 기법: 바이트 페어 인코딩
  
## 2-2. 바이트 페어 인코딩(BPE)이란?
- GPT - BPE
- BERT - 워드피스
- BPE(빈도)
  - 가장 많이 등장한 문자열을 병합하여 데이터를 압축하는 기법
  - 연속된 두 글자를 한 글자로 병합
  - 사전 크기 증가를 억제하면서도 정보를 효율적으로 압축할 수 있음
  - 분석 대상 언어에 대한 지식이 필요 없음
  - 절차
      1. 어휘 집합 구축: 고빈도 바이그램 쌍을 병합하는 방식으로 구축
        - 프리토크나이즈: 말뭉치 공백으로 분절
        - 문자 단위 초기 어휘집합 생성
        - 프리토크나이즈 결과를 초기 어휘 집합으로 재작성
        - 위 결과를 바이그램으로 묶어서 나열
        - 바이그램 쌍 중 가장 빈도 높은 것 어휘 집합에 추가
        - (반복) 해당 어휘집합으로 빈도표 재작성 -> 바이그램으로 묶어 빈도 가장 높은 것 어휘집합에 추가
          - 어휘 집합이 사용자가 정한 크기가 될 때까지 반복
      2. 토큰화
        - '어휘집합'과 '병합 우선순위'가 있으면 수행 가능
        - 프리토크나이즈 수행
        - 문자 단위 분리 -> 우선 순위 참고하여 병합
        - 병합 대상이 없을때까지 반복
        - 어휘집합에 없는 문자는 미등록 토큰(<unk>)

- 워드피스(우도)
  - 말뭉치에서 자주 등장한 문자열을 토큰으로 인식한다는 점에서 BPE와 본질적으로 유사
  - 문자열 병합 기준에서 BPE와 차이를 보임
    - 단순 빈도 기준이 아닌, 병합 시 말뭉치의 likelihood를 가장 높이는 쌍을 병합
    - 워드피스에서는 병합 대상 전체 후보들 가운데 우도 계산값이 가장 높은 쌍을 합침
    - 따라서 워드피스는 '병합 우선순위' 필요 없이 '어휘집합'만 가지고 토큰화 가능
  - 분석 대상 어절에 '어휘집합'에 있는 서브워드가 포함된 경우 해당 서브워드를 어절에서 분리
    - 단, 후보가 여럿일 경우 가장 긴 서브워드 선택
    - 서브워드 후보가 하나도 없을 시에는 해당 문자열 전체를 미등록 단어로 취급
  

## 2-3. 어휘 집합 구축하기
- Korpora 패키지를 통해 말뭉치 다운로드 가능
- BPE: GPT 계열 모델이 사용하는 토크나이저 기법
  - 단 문자 단위가 아니라 '유니코드 바이트' 수준
    - 전세계 대부분의 글자는 유니코드로 표현할 수 있으므로 미등록 토큰 문제에서 비교적 자유로워짐
    - 한글은 한 글자가 3개의 유니코드 바이트로 표현됨
  - 말뭉치를 유니코드 바이트로 변환 -> 토큰화
  - 어휘집합 구축 -> return: vocab.json, merges
- wordpiece: BERT 계열이 사용하는 토크나이저
  - 어휘집합 구축 -> return: vocab.txt

## 2-4. 토큰화하기
- vocab, merges 파일로 사전학습된 토크나이저 사용 가능(BERT는 vocab만)
- GPT 토크나이저 
  - 인풋: 문장, 패딩 기준, 토큰 기준 최대 길이, 문장 잘림 허용 옵션(truncation)
  - 아웃풋: input_ids(토큰 인덱스), attention_mask(일반 토큰과 패딩 토큰 구분)
- BERT 토크나이저
  - 인풋: 문장, 패딩 기준, 토큰 기준 최대 길이, 문장 잘림 허용 옵션(truncation)
  - 아웃풋: input_ids(토큰 인덱스), attention_mask(일반 토큰과 패딩 토큰 구분), token_type_ids(세그먼트 정보, 첫번째문장 = 0, 두번째 문장 = 1, ...)
    - 세그먼트 정보 입력은 BERT 모델의 특징

# 3. 숫자 세계로 떠난 자연어
## 3-1. 미리 학습된 언어모델
- 언어모델: 단어 시퀀스에 확률을 부여하는 모델 / 이전 단어들이 주어졌을 때 다음 단어가 나타날 확률을 부여하는 모델
- 순방향 언어모델(forward language model) - GPT, ELMo의 사전학습 방법
- 역방향 언어모델(backward language model) - ELMo의 사전학습 방법
- 넓은 의미의 언어모델: 컨텍스트가 전제된 상태에서 특정 단어가 나타날 조건부 확률
- 마스크 언어 모델엔 양방향 성질이 있음
- 스킵 그램 모델
  - 어떤 단어 앞뒤에 특정 범위를 정해둠
  - 이 범위 내에 어떤 단어들이 올지 분류
  - 컨텍스트로 설정한 단어 주변에 어떤 단어들이 분포해 있는지 학습
  - Word2Vec이 해당 모델 방식으로 학습
- 언어 모델이 주목받는 이유
  1. '다음단어맞히기', '빈칸맞히기'를 사용하면 레이블링 없이 학습데이터를 싼 값에 만들어 낼 수 있음
  2. 대량의 말뭉치로 언어모델의 프리트레인하면 다운스트림 태스크에서 적은 양의 데이터로 성능 큰 폭 향상 가능

## 3-2. 트랜스포머 살펴보기
- 트랜스포머는 sequence-to-sequence 모델
  - 특정 속성을 지닌 시퀀스를 다른 속성의 시퀀스로 변환하는 작업
  - 임의의 시퀀스를 해당 시퀀스와 속성이 다른 시퀀스로 변환하는 작업은 기계번역이 아니더라도 수행 가능
    - (ex. 기온의 시퀀스 -> 태풍 발생 여부의 시퀀스)
  - 인코더 - 디코더 2개 파트로 구성
- 인코더의 입력: 소스시퀀스 전체
- 디코더의 입력: 문맥(=인코더의 출력), 맞혀야할 단어 이전의 정답 타깃 시퀀스(훈련 시)/이전 디코더 출력(인퍼런스 시)
- 인코더와 디코더 차이: 마스크 멀티 헤드 어텐션, 인코더의 컨텍스트 반영
- 트랜스포머의 경쟁력은 셀프어텐션에 있다!
  - 어텐션: 중요한 요소에 집중하고 그렇지 않은 요소는 무시하기
  - 셀프어텐션: 입력 시퀀스 가운데 의미 있는 요소들 위주로 정보를 추출
    - CNN과 비교: CNN은 합성곱 필터 크기를 넘어서는 문맥은 읽어내기 어려움
    - RNN과 비교: 길이가 길어질수록 정보 압축 어려움. 오래전 입력 단어 잊거나 특정 정보 과도하게 반영(문장의 마지막 단어가 출력에 많이 반영됨)
    - 어텐션과 비교: 디코더 쪽 RNN에 어텐션을 추가. 디코더가 타깃 시퀀스를 생성할 때 소스 시퀀스 전체에서 어떤 요소에 주목해야 할 지를 알려줌
      - 어텐션은 소스 시퀀스 전체 단어들과 타깃 시퀀스 단어 하나 사이를 연결. 반면 셀프 어텐션은 입력 시퀀스 전체 단어들 사이를 연결
      - 셀프어텐션은 RNN 없이 동작
      - 어텐션은 타깃 언어의 단어 1개 생성 시 1회 수행, 셀프 어텐션은 인코더, 디코더 블록의 개수만큼 반복 수행
  

## 3-3. 셀프 어텐션 동작 원리
- 인코더 입력: 입력 입베딩 + 위치 정보(positional encoding)
- 트랜스포머 전체 구조의 출력층: 디코더 마지막 블록 출력 벡터 시퀀스(타깃 언어의 어휘 수만큼의 차원을 갖는 벡터)
  - 타깃 언어의 어휘가 3만개이면. 이 벡터의 차원 수는 3만. 그 3만 개 요솟값을 모두 더하면 합은 1
- 마스크 시 정답을 포함한 타깃 시퀀스의 미래 정보를 마스킹 함
  - 마스킹은 소프트맥스 확률이 0이 되도록 함
  - 즉 디코더는 이전 입력 단어와 인코더에서 출력된 문맥 관계를 가지고 다음 입력 단어를 맞추는 것


## 3-4. 트랜스포머에 적용된 기술들
- 활성 함수는 현재 계산하고 있는 뉴런의 출력을 일정 범위로 제한하는 역할을 함
- 트랜스포머에서는 은닉층의 뉴런 개수를 입력층의 4배로 설정
  - 입력벡터가 768차원일 경우, 은닉층을 2048차원까지 늘렸다가 다시 768차원으로 줄임
- 잔차연결(residual connection)을 통해 여러가지 경로의 학습이 가능 => 다양한 관점에서 블록 계산을 수행
  - 또한 블록을 건너뛰는 경로를 설정함으로써 학습을 쉽게하는 효과까지 거둠
- 레이어 정규화: 미니배치의 인스턴스별로 평균을 빼주고 표준편차로 나눠 정규화를 수행하는 기법
  - 학습이 안정되고 속도가 빨라지는 효과
  - 평균을 빼고 표준편차로 나눈 결과값에 감마를 곱하고, 베타를 더함
    - 이 2가지 하이퍼파라미터는 학습 과정에서 업데이트 되는 가중치. 초깃값은 각각 1, 0
  

## 3-5. BERT와 GPT 비교
- GPT 
  - 언어 모델
  - 트랜스포머의 디코더만 취함
  - 앞 문맥만 활용 가능
- BERT
  - 마스크 언어 모델
  - 트랜스포머의 인코더만 취함
  - 앞 뒤 문맥 모두 활용 가능
- 모델 성능을 최대한 유지하면서 계산량 혹은 모델의 크기를 줄이려는 시도: distillaion, quatization, pruning, weight sharing
  

## 3-6. 단어/문장을 벡터로 변환하기
- BERT 모델의 입력값을 만들려면 토크나이저부터 선언해 두어야 함
- 프리트레인 할 때 썼던 토크나이저를 그대로 사용해야 벡터 변환에 문제가 없음
- 파이토치의 입력값 자료형은 파이토치에서 제공하는 텐서여야 함
- model output의 속성에 last_hidden_state, pooler_output 존재


# 4. 문서에 꼬리표 달기

## 4-1. 문서 분류 모델 훑어보기
- CLS 토큰을 pooler에 입력
- pooler_output에 드랍아웃 적용
- 이후 가중치를 곱해 아웃풋 개수의 출력이 나오도록 변환(감성분석의 경우 2개)
- 소프트맥수 함수에 입력 => 최종출력

## 4-2. 문서 분류 모델 학습하기
- 문서분류 모델 실습(kcbert-base모델을 NSMC 데이터로 파인튜닝)
- TrainArguments의 각 인자 의미
  ```python
  args = ClassificationTrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_name="nsmc",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-doccls",
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    max_seq_length=128,
    epochs=3,
    tpu_cores=0 if torch.cuda.is_available() else 8,
    seed=7,
    )
  ```
  - pretrained_model_name: 사전학습된 언어 이름. 단 허깅페이스 모델 허브에 등록되어 있어야 함
  - downstream_corpus_name: 다운스트림 데이터의 이름
  - downstream_corpus_root_dir: 다운스트림 데이터를 내려받을 위치. 입력하지 않으면 `/root/Korpora`에 저장됨
  - batch_size: 배치 크기. GPU를 선택했다면 32, TPU라면 4.
  
- 데이터 로더는 데이터셋이 보유하고 있는 인스턴스를 배치 크기만큼 뽑아서 자료형, 데이터 길이 등 정해진 형식에 맞춰 배치를 만들어줌
- Dataset은 corpus의 문장과 레이블을 각각 tokenizer를 활영하여, 모델이 학습할 수 있는 형태(ClassificationFeatures; 4가지 정보)로 가공함
  - input_ids: 인덱스로 변환된 토큰 시퀀스
  - attention_mask: 해당 토큰이 패딩인지 아닌지
  - token_type_ids: 세그먼트 정보(문서 구분); BERT모델의 특징; '이어진 문서인지 맞히기'
  - label

- 전체 인스턴스 중 배치 크깅에서 정의한만큼을 뽑아 Dataloader는 배치 형태로 가공(nlpbook.data_collator)
- DataLoader에서 주목할 인자 2개
  - `sampler=RandomSampler(train_dataset, replacement=False)`
    - replacement는 복원추출 여부
  - `collate_fn=nlpbook.data_collator`
    - 뽑은 인스턴스를 배치로 만드는 역할
    - 같은 배치에서 인스턴스가 여럿일 때, 이를 input_ids, attention_mask 등 종류별로 모으고 텐서로 변경
- 평가용(validation) 데이터 로더의 경우, SequentialSampler를 사용함
    - `sampler=SequentialSampler(val_dataset`
      - 평가 시에는 평가용 데이터 전체를 사용하므로 굳이 랜덤으로 구성할 이유가 없음

- `BertForSequenceClassification`은 프리트레인을 마친 BERT 모델 위에 문서 분류용 태스크 모듈이 덧붙여진 형태의 모델 클래스
- 허깅페이스에 등록된 모델이라면 별다른 코드 수정 없이 모델명만 바꿔 사용 가능(단, 토크나이저와 동일한 모델을 사용해야 함)

- 태스크에는 모델, 옵티마이저, 학습과정 등이 정의돼 있음
  -  `ClassificationTask`에는 옵티마이저, 러닝 레이트 스케쥴러가 정의되어 있음
    - LR 스케줄러는 ExponentialLR을 사용함

  - ```python
      from transformers import PreTrainedModel
      from transformers.optimization import AdamW
      from ratsnlp.nlpbook.metrics import accuracy
      from pytorch_lightning import LightningModule
      from torch.optim.lr_scheduler import ExponentialLR
      from ratsnlp.nlpbook.classification.arguments import ClassificationTrainArguments

      class ClassificationTask(LightningModule):

          def __init__(self,
                      model: PreTrainedModel,
                      args: ClassificationTrainArguments,
          ):
              super().__init__()
              self.model = model
              self.args = args

          def configure_optimizers(self):
              optimizer = AdamW(self.parameters(), lr=self.args.learning_rate)
              scheduler = ExponentialLR(optimizer, gamma=0.9)
              return {
                  'optimizer': optimizer,
                  'scheduler': scheduler,
              }

          def training_step(self, inputs, batch_idx):
              # outputs: SequenceClassifierOutput
              outputs = self.model(**inputs)
              preds = outputs.logits.argmax(dim=-1)
              labels = inputs["labels"]
              acc = accuracy(preds, labels)
              self.log("loss", outputs.loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
              self.log("acc", acc, prog_bar=True, logger=True, on_step=True, on_epoch=False)
              return outputs.loss

          def validation_step(self, inputs, batch_idx):
              # outputs: SequenceClassifierOutput
              outputs = self.model(**inputs)
              preds = outputs.logits.argmax(dim=-1)
              labels = inputs["labels"]
              acc = accuracy(preds, labels)
              self.log("val_loss", outputs.loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
              self.log("val_acc", acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
              return outputs.loss
      ```
      - configure_optimizers: 모델 학습에 필요한 Optimizer와 LR Scheduler를 정의. 다른 것들을 사용하려면 여기서 정의하면 됨
      - train_step: 학습 과정에서 한 개의 미니배치가 입력됐을 때, 손실을 계산하는 과정을 정의
      - validation_step: 평가 과정에서 한 개의 미니배치가 입력됐을 때, 손실을 계산하는 과정을 정의
        - 각 스텝에서는 loss, logit, accuracy 등을 계산
        - loss, accuracy 등의 정보를 log에 남기고 메소드를 종료


## 4-3. 학습 마친 모델을 실전 투입하기
- 인퍼런스
- RoBERTa는 BERT와 비슷하나, 어휘 집합 구축/프리트레인 태스크/학습 말뭉치 선정 등 디테일에서 차이를 보임
  - 성능이 좋아 최근에는 BERT보다 널리 쓰이고 있음


# 5. 문장 쌍 분류하기
## 5-1. 문장 쌍 분류 모델 훑어보기
- 문장 쌍 분류의 대표 예: 자연어 추론(NLI)
  - 문장 쌍 분류란 문장 2개가 주어졌을 때, 해당 문장 사이의 관계가 어떤 범주일지 분류하는 과제: 참/거짓/중립 or 판단불가
  - NLI 모델은 전제와 가설2개 문장을 입력으로, 두 문장의 관계가 어떤 범주일지 확률 출력

- 인풋: [CLS] + 전제 + [SEP] + 가설 + [SEP]
- 아웃풋: [전제에 대해 가설이 참일 확률, 거짓일 확률, 중립일 확률]

## 5-2. 문장 쌍 분류 모델 학습하기
- kcbert-base 모델을 업스테이지가 공개한 KLUE-NLI 데이터로 파인튜닝
- 문장 분류와 문장 쌍 분류는 동일한 Dataset을 활용
  - ClassificationDataset 모듈
    ```python
    train_dataset = ClassificationDataset(
      args=args,
      corpus=corpus,
      tokenizer=tokenizer,
      mode="train",
      )
    ```
  - 위 모듈은 동일한 인자임에도 아웃풋 형태가 약간 다름
  - 첫문장은 CLS, SEP 토큰을 포함하여 토큰 수가 2개 증가
  - 두번째 문장 이후로는 SEP 1개 증가


- PL에서 모델은 태스크에 포함된다!
- trainer의 인자는 3가지: 태스크, 학습 데이터 로더, 평가 데이터 로더

## 5-3. 학습 마친 모델을 실전 투입하기
- 인퍼런스 시 모델 로드 순서
  1. 체크 포인트 로드
  ```python
    fine_tuned_model_ckpt = torch.load(
        args.downstream_model_checkpoint_fpath,
        map_location=torch.device("cpu")
    )
  ```

  2. 모델 설정 로드
  ```python
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
    )
  ```
  3. 모델 초기화
  ```python
  model = BertForSequenceClassification(pretrained_model_config)
  ```
  4. 체크포인트 주입
  ```python
  model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
  ```
  5. 평가 모드로 전환(드롭아웃 등 학습 때만 사용하는 기법들 무효화)
  ```python
  model.eval()
  ```

  - 문장 쌍 분류는 두 문서 사이의 유사도 혹은 관련도를 따지는 검색 모델로도 발전시킬 수 있음!


# 6. 단어에 꼬리표 달기
## 6-1. 개체명 인식 모델 훑어보기
- 개체명 인식: 문장을 토큰화 한 뒤, 토큰 각각에 인명, 지명, 기관명 등 개체명 태그를 붙여주는 과제
- 입력: 토큰 시퀀스. 문장 맨 앞에 [CLS], 맨 뒤 [SEP] 태그 추가
- 출력: [해당 토큰의 각 개체명 태그별 소속될 확률], 개체명이 10개일 때 총 11개 확률 제공('개체명 아님 범주'까지)
  - 입력 토큰 수가 n개라면 n * 11 개의 아웃풋

## 6-2. 개체명 인식 모델 학습하기
- 개체명 레이블링 시 태그의 의미
  - B: 태그의 시작
  - I: B가 아닌 태그
  - O: 개체가 아님

- NERFeatures 구성요소
  - input_ids
  - attention_mask
  - token_type_ids
  - label_ids
    - 분류 가능한 개체명 태그 수 + [CLS], [SEP], [PAD]

- Dataset 모듈 마다 mode가 다를 수 있음
  - ClassificationDataset은 test mode로 validation 수행
  - NERDataset은 val mode로 validation 수행

- 시퀀스 레이블링은 토큰 각각이 각 범주들에 대한 확률을 계산하므로 출력하는 값이 더 많음

## 6-3. 학습 마친 모델을 실전 투입하기
- 인퍼런스 함수 외에 크게 다른부분 없음
- NER 모델을 똑같은 구조로 품사(POS) 데이터로 학습 한다면 '품사 부착 모델'을 구축할 수 있음
- 또한 띄어쓰기 교정 모델 역시 만들 수 있음


# 7. 질문에 답하기
## 7-1. 질의응답 모델 훑어 보기
- 질의응담: 질문에 답을 하는 과제
  - 그 유형은 다양하나 이 책의 실습 예시는 질문의 답을 지문(Context)에서 찾는 것임

- 모델 입력: question, context
  - 입력 형태: [CLS] question [SEP] context [SEP]
- 모델 출력: 입력의 각 토큰이 [정답의 시작일 확률, 정답의 끝일 확률]
  - 정답의 시작이 될 수 있는 토큰 들 중 시작일 확률(끝도 마찬가지)

## 7-2. 질의응답 모델 학습하기
- 새로운 인자
  - doc_stride: 지문에서 몇 개의 토큰을 슬라이딩 해가면서 데이터를 늘릴지 결정

- KorQuADV1Corpus 구성
  - question_text(질문)
  - context_text(지문)
  - answer_text(정답)
  - start_position_character(정답의 시작 인덱스)

- QADataset 구성
  - input_ids: [CLS] 질문 [SEP] 지문 [SEP]
  - attention_mask: 패딩 표시
  - token_type_ids: 질문/지문 구분
  - start_positions: 정답 시작
  - end_positions: 정답 끝

- QA 태스크에서 지문의 길이는 BERT 모델이 처리할 수 있는 max_seq_length 보다 긴 경우가 많음
  - 따라서 질문은 그대로 두고 지문 일부를 편집하는 방식으로 학습 데이터를 만듦
  - doc_stride가 64라면, 지문 앞부분 64개 토큰을 없애고 원래 지문 뒷부분에 이어지고 있던 64개 토큰을 가져와 붙임(다음 인스턴스에서)
  - 당연히 정답 시퀀스도 삭제 될 수 있음. 이 경우 start_positions/end_positions는 모두 0


## 7-3. 학습 마친 모델을 실전 투입하기
- 인퍼런스 순서
  - 인자 설정
  - 토크나이저 로드
  - 체크포인트 로드
  - 모델 설정(Config) 로드
  - 모델 초기화
  - 체크포인트 주입
  - 평가모드
  - 출력값 생성 및 후처리