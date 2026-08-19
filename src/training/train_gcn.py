from pathlib import Path
import torch
import torch.nn as nn
from src.models.gcn_model import DrugMicrobeGCN

DATA_FILE = Path('data/graph/model_inputs.pt')
MODEL_FILE = Path('data/graph/best_gcn_model.pt')

EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 15


def evaluate(model, data, split):
    model.eval()
    with torch.no_grad():
        z = model.encode(data['x'], data['edge_index'])
        logits = model.decode(
            z,
            data[f'{split}_drug_id'],
            data[f'{split}_microbe_id']
        )
        labels = data[f'{split}_labels']
        loss = nn.BCEWithLogitsLoss()(logits, labels)
    return loss.item()


def main():
    print('=' * 70)
    print('STEP 107 - GCN TRAINING')
    print('=' * 70)

    data = torch.load(DATA_FILE, weights_only=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    model = DrugMicrobeGCN(
        in_channels=data['x'].shape[1],
        hidden_channels=128,
        embedding_dim=64,
        dropout=0.2,
    ).to(device)

    x = data['x'].to(device)
    edge_index = data['edge_index'].to(device)

    train_drug = data['train_drug_id'].to(device)
    train_microbe = data['train_microbe_id'].to(device)
    train_labels = data['train_labels'].to(device)

    val_drug = data['val_drug_id'].to(device)
    val_microbe = data['val_microbe_id'].to(device)
    val_labels = data['val_labels'].to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(x, edge_index)
        logits = model.decode(z, train_drug, train_microbe)
        train_loss = criterion(logits, train_labels)

        train_loss.backward()
        optimizer.step()

        val_loss = evaluate(
            model,
            {
                'x': x,
                'edge_index': edge_index,
                'val_drug_id': val_drug,
                'val_microbe_id': val_microbe,
                'val_labels': val_labels,
            },
            'val',
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss.item(),
                    'val_loss': val_loss,
                    'in_channels': data['x'].shape[1],
                    'hidden_channels': 128,
                    'embedding_dim': 64,
                    'dropout': 0.2,
                },
                MODEL_FILE,
            )
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f'Epoch {epoch:03d} | '
                f'Train Loss: {train_loss.item():.6f} | '
                f'Val Loss: {val_loss:.6f}'
            )

        if patience_counter >= PATIENCE:
            print(f'Early stopping at epoch {epoch}')
            break

    print()
    print('=' * 70)
    print('TRAINING COMPLETE')
    print('=' * 70)
    print('Best epoch:', best_epoch)
    print('Best validation loss:', f'{best_val_loss:.6f}')
    print('Model saved:', MODEL_FILE)
    print('STEP 107 STATUS: PASSED')


if __name__ == '__main__':
    main()
