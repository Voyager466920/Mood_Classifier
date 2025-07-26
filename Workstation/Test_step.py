import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def test_step(model, dataloader, device, loss_fn, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device).float()

            outputs = model(input_ids, lengths)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = loss_fn(logits, labels)
            running_loss += loss.item() * input_ids.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    avg_loss = running_loss / len(dataloader.dataset)
    y_true = torch.cat(all_labels, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return avg_loss, precision, recall, f1
