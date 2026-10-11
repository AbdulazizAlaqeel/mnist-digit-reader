import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)

# Part 1: load the data
transform = transforms.ToTensor()
train_data = datasets.MNIST("data", train=True, transform=transform)
test_data = datasets.MNIST("data", train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

print("Training pictures:", len(train_data))
print("Test pictures:", len(test_data))

# Part 2: define the model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

print(model)

# Part 3: train the model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10):
    for pictures, labels in train_loader:
        scores = model(pictures)
        loss = loss_fn(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} done, last loss: {loss.item():.3f}")

    # Part 4: test the model
model.eval()
correct = 0
with torch.no_grad():
    for pictures, labels in test_loader:
        scores = model(pictures)
        guesses = scores.argmax(dim=1)
        correct += (guesses == labels).sum().item()

accuracy = 100 * correct / len(test_data)
print(f"Test accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "mnist_model.pt")
print("Model saved to mnist_model.pt")