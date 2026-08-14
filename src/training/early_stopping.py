import torch


class EarlyStopping:

    def __init__(
        self,
        patience=20,
        min_delta=0.0,
        save_path="saved_models/best_model.pth"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path

        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(
        self,
        val_loss,
        model,
        decoder,
        optimizer,
        epoch
    ):

        # --------------------------------
        # Check Improvement
        # --------------------------------

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss
            self.counter = 0

            # --------------------------------
            # Save Best Checkpoint
            # --------------------------------

            torch.save(
                {
                    "epoch": epoch,
                    "loss": val_loss,
                    "gcn_state_dict": model.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                self.save_path
            )

            print(
                f"Validation improved. "
                f"Best Loss = {val_loss:.6f}"
            )

            return False

        # --------------------------------
        # No Improvement
        # --------------------------------

        self.counter += 1

        print(
            f"No improvement: "
            f"{self.counter}/{self.patience}"
        )

        if self.counter >= self.patience:

            self.early_stop = True

            print(
                "\nEarly stopping triggered."
            )

            return True

        return False