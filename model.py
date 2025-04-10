# model.py

import os
import json
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,TrainingArguments,Trainer,EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_and_save_model(
    tokenized_dataset,
    tokenizer,
    model_name="bert-base-uncased",
    output_dir="./model_output",
    num_epochs=10,
    batch_size=16,
    learning_rate=2e-5,
    early_stopping_patience=2
):
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    print(f"載入預訓練模型 {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    ).to(device)

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/checkpoints",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )

    print("開始訓練模型")
    trainer.train()

    print("評估模型")
    eval_results = trainer.evaluate()
    print("評估結果")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")

    model_save_path = f"{output_dir}/final_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"模型訓練完成並保存至: {model_save_path}")

    
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    return model, tokenizer, eval_results