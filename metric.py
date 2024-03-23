import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

hf_token = "blank"

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


def load_data(filepath):
    df = pd.read_csv(filepath)
    # Concatenate 'american_text' and 'german_text' into 'text' column
    df['text'] = df['american_text'].fillna('') + ' ' + df['german_text'].fillna('')
    # Create a label column (1 for German, 0 for American)
    df['label'] = df.apply(lambda x: 1 if pd.notna(x['german_text']) else 0, axis=1)
    return df[['text', 'label']]



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    # Load and preprocess the training data
    training_filepath = 'american_german.csv'
    df = load_data(training_filepath)
    # Optionally, reduce the size of the dataset for faster training
    df = df.sample(frac=0.01)  # Adjust the fraction as needed

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # Initialize the tokenizer and model (using DistilBERT)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Freeze all layers except the classifier layer
    for param in model.distilbert.parameters():
        param.requires_grad = False

    # Tokenize data
    train_encodings = tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=256)
    test_encodings = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=256)

    # Create torch datasets
    train_dataset = CustomDataset(train_encodings, y_train.tolist())
    test_dataset = CustomDataset(test_encodings, y_test.tolist())

    # Define training arguments
    training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=32, evaluation_strategy="epoch")

    # Initialize Trainer
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)

    # Train the model
    trainer.train()

    # Evaluation
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))


if __name__ == "__main__":
    main()
