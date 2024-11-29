import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load your additional dataset (Assuming it's in CSV format)
def load_custom_dataset(file_path):
    # Replace with your dataset loading logic (CSV, JSON, etc.)
    #dataset = Dataset.from_jsonl(file_path)
    dataset = load_dataset('json', data_files=file_path)
    # Ensure the dataset has 'premise', 'hypothesis', 'label' columns
    return dataset

# Preprocess the dataset
def preprocess_dataset(tokenizer, dataset, max_length=128):
    def tokenize_function(examples):
        return tokenizer(
            examples["premise"], 
            examples["hypothesis"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
    
    return dataset.map(tokenize_function, batched=True)

# Compute metrics for evaluation
def compute_metrics(pred):
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

# Main fine-tuning script
def fine_tune_electra(base_model_path, custom_dataset_path, output_dir, epochs=3, batch_size=32):
    # Load the tokenizer and model
    tokenizer = ElectraTokenizer.from_pretrained(base_model_path)
    model = ElectraForSequenceClassification.from_pretrained(base_model_path, num_labels=3)  # 3 for NLI
    
    # Load and preprocess the dataset
    dataset = load_custom_dataset(custom_dataset_path)
    dataset = preprocess_dataset(tokenizer, dataset)
    #dataset = dataset.train_test_split(test_size=0.01)  # Split into train and validation sets

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        
    )
    
    # Create the Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["train"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Fine-tune the model
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")

# Fine-tune the model (Adjust paths and parameters as necessary)
fine_tune_electra(
    base_model_path="C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\fp-dataset-artifacts\\trained_model\\",  # Pretrained model path
    custom_dataset_path="C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\snli_negation_test.jsonl",  # Path to your additional dataset
    output_dir="C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\fp-dataset-artifacts\\retrained_output\\electra_fine_tuned",  # Directory to save the fine-tuned model
    epochs=3,
    batch_size=32
)
