"""Fine-tune ResNet-32 for jersey numbers, reproducing the blog's winner.

blog.roboflow.com/identify-basketball-players trains two number readers on
basketball-jersey-numbers-ocr/3 and concludes: "ResNet-32 ... reached 93%
test accuracy, outperforming the fine-tuned SmolVLM2" (86%). The blog
publishes the dataset and the conclusion but not the training code, so
provenance per choice, stated plainly:

  from the blog     the dataset (JSONL, prefix "Read the number."), the
                    reformat "from JSONL into a directory-structured
                    dataset typical for image classification", the
                    architecture name ResNet-32, "fine-tuned"
  filled in here    ResNet-32 means the CIFAR-style 32-layer net (He et
                    al. 2015) -- taken pretrained on CIFAR-100 from
                    torch.hub chenyaofo/pytorch-cifar-models, which is
                    what fine-tuning a "ResNet-32" can concretely mean;
                    32x32 input (the architecture's native size); SGD
                    momentum 0.9, lr 0.05 cosine to 60 epochs, batch 128,
                    weight decay 5e-4; colour jitter + small random crop,
                    and NO horizontal flip -- digits are chiral

    python pipeline/train_resnet_ocr.py --prepare   # JSONL -> class dirs
    python pipeline/train_resnet_ocr.py --train
    python pipeline/train_resnet_ocr.py --eval      # test accuracy

Rows whose test label never occurs in training count as errors, not
exclusions -- the model cannot answer them and pretending otherwise
inflates the score.
"""

import argparse
import json
import shutil
from pathlib import Path

DS = Path("out/jersey_ocr_ds")
CLS = Path("out/jersey_cls")
CKPT = Path("out/resnet32_jersey.pt")
EPOCHS = 60
BATCH = 128
LR = 0.05
SIZE = 32


def rows(split):
    for line in (DS / split / "annotations.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("prefix") == "Read the number." and r.get("suffix", "").strip().isdigit():
            yield r["image"], r["suffix"].strip()


def prepare():
    for split in ("train", "valid", "test"):
        n = 0
        for image, label in rows(split):
            dst = CLS / split / label
            dst.mkdir(parents=True, exist_ok=True)
            src = DS / split / image
            if src.exists():
                shutil.copy(src, dst / image)
                n += 1
        print(f"{split}: {n} labelled crops")


def loaders():
    import torch
    from torchvision import datasets, transforms

    train_tf = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.RandomCrop(SIZE, padding=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
    ])
    train = datasets.ImageFolder(CLS / "train", train_tf)
    valid = datasets.ImageFolder(CLS / "valid", eval_tf)
    dl = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=BATCH, shuffle=sh, num_workers=2, pin_memory=True)
    return train, dl(train, True), valid, dl(valid, False)


def build(num_classes):
    import torch
    model = torch.hub.load("chenyaofo/pytorch-cifar-models",
                           "cifar100_resnet32", pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def train():
    import torch
    from progress import Progress

    train_ds, train_dl, valid_ds, valid_dl = loaders()
    print(f"{len(train_ds)} train crops, {len(train_ds.classes)} number classes, "
          f"{len(valid_ds)} valid")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(len(train_ds.classes)).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = torch.nn.CrossEntropyLoss()

    # valid uses train's class indexing; unseen valid labels are impossible
    remap = {i: train_ds.class_to_idx.get(c, -1)
             for i, c in enumerate(valid_ds.classes)}

    best = 0.0
    prog = Progress("resnet-ocr", total=EPOCHS)
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        hit = tot = 0
        with torch.no_grad():
            for x, y in valid_dl:
                pred = model(x.to(dev)).argmax(1).cpu()
                y = torch.tensor([remap[int(v)] for v in y])
                hit += (pred == y).sum().item()
                tot += len(y)
        acc = hit / tot
        if acc > best:
            best = acc
            torch.save({"state": model.state_dict(),
                        "classes": train_ds.classes}, CKPT)
        prog.step(note=f"epoch {epoch + 1}, valid {acc:.1%} (best {best:.1%})")
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}: valid {acc:.1%}  best {best:.1%}",
                  flush=True)
    prog.done(note=f"best valid {best:.1%}")
    print(f"best valid accuracy {best:.1%}, saved {CKPT}")


def evaluate():
    import torch
    from PIL import Image
    from torchvision import transforms

    ck = torch.load(CKPT, weights_only=False)
    classes = ck["classes"]
    model = build(len(classes))
    model.load_state_dict(ck["state"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])

    hit = tot = unseen = 0
    with torch.no_grad():
        for image, label in rows("test"):
            src = DS / "test" / image
            if not src.exists():
                continue
            x = tf(Image.open(src).convert("RGB")).unsqueeze(0).to(dev)
            pred = classes[int(model(x).argmax(1))]
            tot += 1
            hit += (pred == label)
            unseen += (label not in classes)
    print(f"ResNet-32 test accuracy: {hit}/{tot} = {hit/tot:.1%}")
    print(f"  ({unseen} test crops carry a number never seen in training -- "
          f"counted as errors; the blog reports 93%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare()
    if args.train:
        train()
    if args.eval:
        evaluate()
    if not (args.prepare or args.train or args.eval):
        print("pass --prepare, --train and/or --eval")


if __name__ == "__main__":
    main()
