import torch
import torchvision
from torchvision import transforms

from unet import UNet
from data import create_datasets
from train import loss_fn
from utils import get_device


def export_image(image_batch, filename):
    iamge_grid = torchvision.utils.make_grid(image_batch.to(torch.float), nrow=8)
    image = transforms.ToPILImage()(iamge_grid)
    image.save("output/{}.png".format(filename), "PNG")


def inspect_data():
    train_dataloader, test_dataloader = create_datasets()
    (train_input_batch, train_target_batch) = next(iter(train_dataloader))
    (test_input_batch, test_target_batch) = next(iter(test_dataloader))
    print(train_input_batch.shape, train_target_batch.shape)
    print(test_input_batch.shape, test_target_batch.shape)
    print(train_input_batch.device)

    export_image(train_input_batch, "inputs")
    export_image(train_target_batch, "targets")


def test_loss_fn():
    x = torch.rand((1, 3, 1, 1))
    y = torch.randint(0, 3, (1, 1, 1, 1), dtype=torch.long)

    x1 = torch.rand((1, 3, 1, 1))
    x1[:, 0, :, :] += 1000.0
    y[:, :, :, :] = 0
    print(x.shape, x1.shape, y.shape)
    assert loss_fn(x, y) > loss_fn(x1, y)

    x = torch.rand((2, 3, 128, 128), requires_grad=True)
    y = torch.randint(0, 3, (2, 1, 128, 128), dtype=torch.long)
    print(loss_fn(x, y))


def inference():
    model = UNet(3).to(get_device())
    model.load_state_dict(
        torch.load("checkpoint_1.pth", map_location=torch.device(get_device()))
    )

    _, test_dataloader = create_datasets()
    (test_input_batch, test_target_batch) = next(iter(test_dataloader))
    test_input_batch = test_input_batch.to(torch.float)
    print(test_input_batch.device, test_input_batch.shape, test_input_batch.dtype)

    logits = model(test_input_batch)
    predicts = logits.argmax(dim=1)
    predicts = predicts.view(predicts.shape[0], 1, predicts.shape[1], predicts.shape[2])

    print(predicts.shape, test_target_batch.shape)

    export_image(test_input_batch, "inputs")
    export_image(test_target_batch, "targets")
    export_image(predicts, "predicts")


def main():
    # test_loss_fn()
    # inspect_data()
    inference()


if __name__ == "__main__":
    main()
