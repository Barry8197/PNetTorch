import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

def train(model, input_arrray, labels, device, batch_size=64, learning_rate=0.01, lr_decay=0.1, num_epochs=10 , sparse=False):
    X_train, X_test, y_train, y_test = train_test_split(
        input_arrray, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Converting numpy arrays back to tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

    model.to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    if sparse : 
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else :
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=lr_decay)

    model.train()
    with tqdm(range(num_epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            running_loss = 0
            for data, targets in train_loader:
                tepoch.set_description(f"Epoch {epoch+1}")
                
                optimizer.zero_grad()
                outputs = model(data)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            lr_scheduler.step()
            #tepoch.update(1)
            tepoch.set_postfix(loss=running_loss)
    
    print('Training complete.')
    return model, train_loader, test_loader

def evaluate(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.tolist())
            _, true = torch.max(labels, 1)
            y_true.extend(true.tolist())
    
    report = classification_report(y_true, y_pred, digits=4)
    print(report)