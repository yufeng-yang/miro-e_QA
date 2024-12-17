import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# Load dataset
data = pd.read_csv('C:\\CodeSpace\\integrate_video_audio_features\\fused_features.csv')

# Remap labels to 0-based indexing
label_mapping = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}
data['Label'] = data['Label'].map(label_mapping)

# Separate features and labels
X = data.iloc[:, 1:].values  # All columns except the first one
y = data.iloc[:, 0].values   # The first column is the label

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize model
input_size = X_train.shape[1]
num_classes = 7
model = EmotionClassifier(input_size, num_classes)



import torch.optim as optim

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



from torch.utils.data import DataLoader, TensorDataset

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


from sklearn.metrics import classification_report, accuracy_score

# Evaluate on the test set
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predictions = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, target_names=[
        "Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]))

torch.save(model.state_dict(), 'emotion_classifier.pth')

