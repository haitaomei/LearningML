import torch
import torch.optim as optim

from torch.optim import lr_scheduler
from unet import UNet
from utils import get_device
from data import create_datasets


_CHECKPOINT_PATH = "checkpoint.pth"
_EVAL_EVERY = 30
_LEARNING_RATE = 1e-4


def accuracy_fn(logits, labels):
    logits = logits.argmax(dim=1)
    labels = labels.view(labels.size(0), labels.size(2), labels.size(3))

    equal = labels == logits
    return 100.0 * equal.sum().item() / logits.numel()


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )

    return loss.mean()


# loss function is inspired by https://github.com/dhruvbird/ml-notebooks
def loss_fn(pred, target, bce_weight=0.5):
    target = torch.cat([(target == i) for i in range(3)], dim=1)
    target = target.to(torch.float)

    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


def train_step(model, optimizer, inputs, labels):
    optimizer.zero_grad()

    logits = model(inputs)

    loss = loss_fn(logits, labels)
    loss.backward()

    optimizer.step()

    return loss.item(), accuracy_fn(logits, labels)


def train(model, num_epochs=1):
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=_LEARNING_RATE
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    for epoch in range(num_epochs):
        print("Epoch {}/{}\n{}".format(epoch + 1, num_epochs, "=" * 30))
        train_dataloader, _test_dataloader = create_datasets(batch_size=4)

        for step, batch in enumerate(train_dataloader):
            inputs, labels = batch
            inputs = inputs.to(get_device())
            labels = labels.to(get_device())
            loss, accuracy = train_step(model, optimizer, inputs, labels)

            if step % _EVAL_EVERY == 0:
                print("Step ", step, "\tloss: ", loss, "\t accuracy: ", accuracy)

        scheduler.step()
        for param_group in optimizer.param_groups:
            print("Learning rate", param_group["lr"])

    torch.save(model.state_dict(), _CHECKPOINT_PATH)
    print("model saved")
    return model


if __name__ == "__main__":
    model = UNet(3).to(get_device())
    model = train(model, num_epochs=10)
