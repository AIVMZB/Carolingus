import torch
from model import FullModel
from dataset_loader import get_dataloader
import torch.optim as optim
import torch.nn as nn
from train_utils import evaluate_model, save_training_graphs
from tqdm import tqdm


def main():
    train_dir = "D:\\GitHub\\Latina\\new_dataset_train"
    val_dir = "D:\\GitHub\\Latina\\new_dataset_val"

    batch_size = 16
    num_epochs = 25
    learning_rate = 0.0005
    num_classes = 1765

    train_loader, val_loader = get_dataloader(train_dir, val_dir, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel(num_classes).to(device)
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    ##
    train_classes = train_loader.dataset.label_to_idx
    val_classes = val_loader.dataset.label_to_idx

    print("First 5 train labels:")
    for i, (label, index) in enumerate(train_classes.items()):
        print(f"{label}: {index}")
        if i == 10:
            break

    print("\nFirst 5 val labels:")
    for i, (label, index) in enumerate(val_classes.items()):
        print(f"{label}: {index}")
        if i == 10:
            break
##
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        tp = 0
        fp = 0
        fn = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
        train_recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0.0

        val_loss, val_accuracy, val_precision, val_recall = evaluate_model(model, val_loader, criterion, device)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Precision: {train_precision:.2f}%, Recall: {train_recall:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, Precision: {val_precision:.2f}%, Recall: {val_recall:.2f}%")


    torch.save(model.state_dict(), "model_3.pth")
    save_training_graphs(train_losses, val_losses, train_accuracies, val_accuracies)


if __name__ == '__main__':
    main()
