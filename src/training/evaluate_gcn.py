from pathlib import Path
import torch
from src.models.gcn_model import DrugMicrobeGCN

data = torch.load(
    'data/graph/model_inputs.pt',
    weights_only=False
)

ckpt = torch.load(
    'data/graph/best_gcn_model.pt',
    weights_only=False
)

model = DrugMicrobeGCN(
    in_channels=ckpt['in_channels'],
    hidden_channels=ckpt['hidden_channels'],
    embedding_dim=ckpt['embedding_dim'],
    dropout=ckpt['dropout']
)

model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print('=' * 70)
print('STEP 108 - BEST MODEL EVALUATION PIPELINE')
print('=' * 70)

print('Best epoch:', ckpt['epoch'])
print('Saved train loss:', f"{ckpt['train_loss']:.6f}")
print('Saved validation loss:', f"{ckpt['val_loss']:.6f}")
print(
    'Model parameters:',
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)

with torch.no_grad():
    z = model.encode(
        data['x'],
        data['edge_index']
    )

    val_logits = model.decode(
        z,
        data['val_drug_id'],
        data['val_microbe_id']
    )

    test_logits = model.decode(
        z,
        data['test_drug_id'],
        data['test_microbe_id']
    )

    val_probs = torch.sigmoid(val_logits)
    test_probs = torch.sigmoid(test_logits)

out = Path('data/graph')

torch.save(
    {
        'val_logits': val_logits,
        'val_probs': val_probs,
        'val_labels': data['val_labels'],
        'test_logits': test_logits,
        'test_probs': test_probs,
        'test_labels': data['test_labels'],
        'embedding': z
    },
    out / 'evaluation_outputs.pt'
)

print('Node embeddings:', tuple(z.shape))
print('Validation predictions:', tuple(val_probs.shape))
print('Test predictions:', tuple(test_probs.shape))

print(
    'Validation probability range:',
    float(val_probs.min()),
    'to',
    float(val_probs.max())
)

print(
    'Test probability range:',
    float(test_probs.min()),
    'to',
    float(test_probs.max())
)

print(
    'NaN validation:',
    bool(torch.isnan(val_probs).any())
)

print(
    'NaN test:',
    bool(torch.isnan(test_probs).any())
)

print('Saved:', out / 'evaluation_outputs.pt')

status = (
    tuple(z.shape) == (1554, 64)
    and tuple(val_probs.shape) == (742,)
    and tuple(test_probs.shape) == (740,)
    and not bool(torch.isnan(val_probs).any())
    and not bool(torch.isnan(test_probs).any())
)

print(
    'STEP 108 STATUS:',
    'PASSED' if status else 'FAILED'
)
