import numpy as np
import tensorflow as tf
import pickle
from transformers import AutoTokenizer

# 1) TFLite 인터프리터 로드
interpreter = tf.lite.Interpreter(model_path="model_kote.tflite")
interpreter.allocate_tensors()

# 입/출력 텐서 정보 확인
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2) 토크나이저·MLB 로드
checkpoint = "monologg/koelectra-small-v3-discriminator"
tokenizer  = AutoTokenizer.from_pretrained(checkpoint)

with open("Checkpoints/mlb.pkl", "rb") as f:
    mlb = pickle.load(f)  # sklearn.preprocessing.MultiLabelBinarizer

# 3) inference 함수
def tflite_predict(texts, threshold=0.5, max_length=128):
    # 토큰화
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"      # numpy array 반환
    )
    input_ids = enc["input_ids"].astype(np.int32)
    # mask: 1/0 int 형
    mask = enc["attention_mask"].astype(np.int32)

    # tflite 모델에 맞춰 각 입력 텐서에 값 설정
    for det in input_details:
        name = det["name"]
        idx  = det["index"]
        if "input_ids" in name:
            interpreter.set_tensor(idx, input_ids)
        elif "mask" in name or "attention_mask" in name:
            interpreter.set_tensor(idx, mask)
        else:
            raise ValueError(f"Unknown input tensor name: {name}")

    # 실행
    interpreter.invoke()

    # 결과 가져오기
    output_data = interpreter.get_tensor(output_details[0]["index"])  # shape (batch, num_labels)
    probs = 1 / (1 + np.exp(-output_data))  # sigmoid

    # threshold 기준으로 라벨 추출
    results = []
    for row in probs:
        preds = [lbl for lbl, p in zip(mlb.classes_, row) if p > threshold]
        results.append(preds)
    return results

# 4) 테스트
samples = [
    "오늘 기분이 정말 좋네요!",
    "아무것도 하기 싫고 우울해요..."
]
print(tflite_predict(samples))
