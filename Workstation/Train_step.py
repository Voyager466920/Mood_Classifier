def train_step(model, device, dataloader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        lengths   = batch["lengths"].to(device)       # attention_mask 대신
        labels    = batch["labels"].to(device)

        # LSTMClassifier는 (input_ids, lengths)만 받도록 구현했으니
        logits = model(input_ids, lengths)

        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_ids.size(0)

    return running_loss / len(dataloader.dataset)
