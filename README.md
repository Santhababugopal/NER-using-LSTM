# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Named Entity Recognition (NER) is a sequence labeling task in Natural Language Processing where each word in a sentence is classified into predefined categories such as:

PER – Person

LOC – Location

ORG – Organization

O – Outside any entity

For this implementation, a standard NER dataset such as the CoNLL-2003 dataset is used.
The dataset contains labeled sentences where each word is tagged with its corresponding entity label.

<img width="347" height="713" alt="image" src="https://github.com/user-attachments/assets/30329d28-a929-47fd-b3bb-74c56d34b26b" />

## DESIGN STEPS

STEP 1:
Import necessary libraries.

STEP 2:
Load dataset , Read and clean the input data.

STEP 3:
Structure data into sentences with word-tag pairs.

STEP 4:
Convert words and tags to indices using vocab dictionaries.

STEP 5:
Pad sequences, convert to tensors, and batch them.

STEP 6:
Create a model with Embedding, BiLSTM, and Linear layers.

STEP 7:
Use training data to update model weights with loss and optimizer.

STEP 8:
Check performance on validation data after each epoch.

STEP 9:
Display predictions or plot loss curves.



## PROGRAM
### Name:SANTHABABU G
### Register Number:212224040292
```
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=256):
        super(BiLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        lstm_out, _ = self.lstm(embeddings)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space


model = BiLSTMTagger(len(words) + 1, len(tags)).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=tag2idx["O"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
      model.train()
      total_loss = 0
      for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      train_losses.append(total_loss)

      model.eval()
      val_loss = 0
      with torch.no_grad():
        for batch in test_loader:
          input_ids = batch["input_ids"].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids)
          loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
          val_loss += loss.item()
      val_losses.append(val_loss)
      print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")          
    return train_losses, val_losses


```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot


<img width="847" height="584" alt="image" src="https://github.com/user-attachments/assets/1ddb8ec6-c7fd-4919-adc5-97cf77dc43c4" />


### Sample Text Prediction

<img width="471" height="523" alt="image" src="https://github.com/user-attachments/assets/c06272fd-587e-43cf-8c9a-e1b9a4db757e" />

## RESULT

Thus the LSTM-based Named Entity Recognition (NER) model was successfully developed and trained.

