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
class BiLSTM(nn.Module):
  def __init__(self, vocab_size,tagset_size, embedding_dim=50, hidden_dim=100):
    super(BiLSTM,self).__init__()
    self.embedding=nn.Embedding(vocab_size,embedding_dim)
    self.dropout=nn.Dropout(0.1)
    self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
    self.fc=nn.Linear(hidden_dim*2,tagset_size)

  def forward(self, input_ids):
      x = self.embedding(input_ids)
      x = self.dropout(x)
      x, _ = self.lstm(x)
      return self.fc(x)



model =BiLSTM(len(word2idx)+1,len(tag2idx)).to(device)
criterion =nn.CrossEntropyLoss()
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


<img width="820" height="564" alt="image" src="https://github.com/user-attachments/assets/2c4b2658-d87d-4e5c-b204-99ea6eef1c5d" />

### Sample Text Prediction

<img width="395" height="509" alt="image" src="https://github.com/user-attachments/assets/ee5a30d5-bca6-48e6-8fb4-a1225240f4b5" />

## RESULT

Thus the LSTM-based Named Entity Recognition (NER) model was successfully developed and trained.

