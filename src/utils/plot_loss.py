import matplotlib.pyplot as plt


def plot_loss(losses, save_path="results/loss_curve.png"):

    plt.figure(figsize=(8,5))

    plt.plot(losses, marker="o")

    plt.title("Training Loss Curve")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()