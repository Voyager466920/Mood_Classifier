import json
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# 1) 데이터 준비
repo_dir = snapshot_download("searle-j/kote", repo_type="dataset")
with open(f"{repo_dir}/raw.json", encoding="utf-8") as f:
    raw = json.load(f)
records = []
for obj in raw.values():
    labels = set()
    for r in obj["labels"].values():
        labels.update(r)
    records.append({"text": obj["text"], "labels": list(labels)})
ds = Dataset.from_list(records).train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = ds["train"], ds["test"]

# 2) 토크나이징
checkpoint = "monologg/koelectra-small-v3-discriminator"
tokenizer  = AutoTokenizer.from_pretrained(checkpoint)

def encode(example):
    toks = tokenizer(
        example["text"],
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    # 토큰은 리스트로 반환해야 HuggingFace Datasets에서 동작
    return {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
        "labels": example["labels"]
    }

train_ds = train_ds.map(encode, batched=False)
val_ds   = val_ds.map(encode, batched=False)

# 3) MultiLabelBinarizer
mlb = MultiLabelBinarizer().fit(train_ds["labels"])

def binarize_labels(example):
    # labels를 binary 벡터로 변환
    example["label_vec"] = mlb.transform([example["labels"]])[0].astype("float32")
    return example

train_ds = train_ds.map(binarize_labels)
val_ds = val_ds.map(binarize_labels)

num_labels = len(mlb.classes_)

# 4) TensorFlow dataset 변환 (datasets.Dataset의 to_tf_dataset 미사용)
def to_tf_dataset(hf_ds, batch_size=128, shuffle=False):
    def gen():
        for ex in hf_ds:
            yield (
                {
                    "input_ids": ex["input_ids"],
                    "attention_mask": ex["attention_mask"],
                },
                ex["label_vec"],
            )
    out_types = (
        {
            "input_ids": tf.int32,
            "attention_mask": tf.int32,
        },
        tf.float32,
    )
    out_shapes = (
        {
            "input_ids": (128,),
            "attention_mask": (128,),
        },
        (num_labels,),
    )
    ds = tf.data.Dataset.from_generator(
        gen, output_types=out_types, output_shapes=out_shapes
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=1024)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_tf = to_tf_dataset(train_ds, shuffle=True)
val_tf = to_tf_dataset(val_ds, shuffle=False)

# 5) 모델 정의 (mask_zero 사용, LSTM에 mask전달 방식 변경)
class KerasLSTM(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.lstm  = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_dim, return_sequences=False, dropout=0.5)
        )
        self.out   = tf.keras.layers.Dense(num_labels, activation="sigmoid")

    def call(self, inputs):
        x = self.embed(inputs["input_ids"])
        # mask 인풋 모양 맞추기
        mask = tf.cast(tf.not_equal(inputs["input_ids"], 0), tf.bool)
        x = self.lstm(x, mask=mask)
        return self.out(x)

vocab_size = tokenizer.vocab_size
model = KerasLSTM(vocab_size, 128, 256, num_labels)
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(1e-3, decay=1e-5),
    metrics=[
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.BinaryAccuracy()
    ]
)

# 6) 훈련
model.fit(train_tf, validation_data=val_tf, epochs=10)

# 7) TFLite 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("model_kote.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as model_kote.tflite")
