import torch
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F


def evaluate_model(model, dataloader, criterion, device):
    model.eval() 
    running_loss = 0.0
    correct = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    val_loss = running_loss / len(dataloader)
    val_accuracy = 100 * correct / total
    precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return val_loss, val_accuracy, precision, recall


def save_training_graphs(train_losses, val_losses, train_accuracies, val_accuracies, output_path="training_graphs_3.png"):
    
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
