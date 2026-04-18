import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage1_ocr.model import OCRNet, EMNIST_LABELS

BATCH_SIZE   = 128
EPOCHS       = 20
LR           = 0.001
NUM_CLASSES  = 62
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "best.pt")


def add_gaussian_noise(images, std_range=(0.1, 0.4)):
    std = torch.empty(1).uniform_(*std_range).item()
    return torch.clamp(images + torch.randn_like(images) * std, 0.0, 1.0)


def add_salt_and_pepper(images, amount_range=(0.02, 0.08)):
    amount = torch.empty(1).uniform_(*amount_range).item()
    noisy  = images.clone()
    rand   = torch.rand_like(images)
    noisy[rand < amount / 2]     = 1.0
    noisy[rand > 1 - amount / 2] = 0.0
    return noisy


def apply_random_noise(images):
    choice = torch.randint(0, 3, (1,)).item()
    if choice == 1:
        return add_gaussian_noise(images)
    if choice == 2:
        return add_salt_and_pepper(images)
    return images


class _TransposeEMNIST:
    def __call__(self, x):
        return x.transpose(1, 2)


def load_emnist():
    transform = transforms.Compose([transforms.ToTensor(), _TransposeEMNIST()])
    print("Loading EMNIST ...")
    kw = dict(root="./data", split="byclass", download=True, transform=transform)
    tr_ds = torchvision.datasets.EMNIST(train=True,  **kw)
    te_ds = torchvision.datasets.EMNIST(train=False, **kw)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Train: {len(tr_ds):,}  |  Test: {len(te_ds):,}")
    return tr_loader, te_loader


def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = correct = total = 0
    t0 = time.time()
    for i, (images, labels) in enumerate(loader):
        images = apply_random_noise(images).to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = outputs.max(1)
        total   += labels.size(0)
        correct += pred.eq(labels).sum().item()
        if (i + 1) % 200 == 0:
            print(f"  E{epoch} [{i+1}/{len(loader)}]  loss={total_loss/(i+1):.4f}  acc={100.*correct/total:.1f}%  ({time.time()-t0:.0f}s)")
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, noise_type=None):
    model.eval()
    total_loss = correct = total = 0
    for images, labels in loader:
        if noise_type == "gaussian":
            images = add_gaussian_noise(images, (0.3, 0.3))
        elif noise_type == "salt_pepper":
            images = add_salt_and_pepper(images, (0.05, 0.05))
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        out = model(images)
        total_loss += criterion(out, labels).item()
        _, pred = out.max(1)
        total   += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return total_loss / len(loader), 100. * correct / total


def main():
    print("=" * 60)
    print(f"OCRNet Training  |  device={DEVICE}")
    print("=" * 60)

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    tr_loader, te_loader = load_emnist()

    model     = OCRNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, tr_loader, criterion, optimizer, epoch)
        te_loss, te_acc = evaluate(model, te_loader, criterion)
        scheduler.step(te_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch}/{EPOCHS}  ({time.time()-t0:.0f}s)")
        print(f"  train  loss={tr_loss:.4f}  acc={tr_acc:.1f}%")
        print(f"  test   loss={te_loss:.4f}  acc={te_acc:.1f}%  lr={lr:.6f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                "arch":        "ocr_net",
                "state":       model.state_dict(),
                "alphabet":    EMNIST_LABELS,
                "num_classes": NUM_CLASSES,
                "epoch":       epoch,
                "accuracy":    te_acc,
            }, WEIGHTS_PATH)
            print(f"  saved best.pt  acc={te_acc:.1f}%")
        print("-" * 60)

    print("\n=== Noise evaluation ===")
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["state"])

    _, clean_acc = evaluate(model, te_loader, criterion)
    _, gauss_acc = evaluate(model, te_loader, criterion, noise_type="gaussian")
    _, sp_acc    = evaluate(model, te_loader, criterion, noise_type="salt_pepper")

    print(f"  Clean:             {clean_acc:.2f}%")
    print(f"  Gaussian (σ=0.3):  {gauss_acc:.2f}%")
    print(f"  Salt&Pepper (5%):  {sp_acc:.2f}%")


if __name__ == "__main__":
    main()
