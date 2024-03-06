import os
import matplotlib.pyplot as plt
from typing import List

def plot_loss_accuracy(
        train_losses: List[int], 
        test_losses: List[int], 
        train_accuracies: List[int], 
        test_accuracies: List[int], 
        save: bool = False,
    ) -> None:   
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, test_losses, label="Testing loss")
    plt.title("Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training accuracy")
    plt.plot(epochs, test_accuracies, label="Testing accuracy")
    plt.title("Accuracy Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    
    if save:
        if not os.path.exists("results"): os.makedirs("results")
        plt.savefig(f"results/AlexNet.png")
    
    plt.show()