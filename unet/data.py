# Copied and modified from https://github.com/dhruvbird/ml-notebooks/

import os

import torch
import torchvision
import torchvision.transforms as T


class OxfordIIITPetsAugmented(torchvision.datasets.OxfordIIITPet):
    def __init__(
        self,
        root: str,
        split: str,
        target_types="segmentation",
        download=False,
        pre_transform=None,
        post_transform=None,
        pre_target_transform=None,
        post_target_transform=None,
        common_transform=None,
    ):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)

        # Common transforms are performed on both the input and the labels
        # by creating a 4 channel image and running the transform on both.
        # Then the segmentation mask (4th channel) is separated out.
        if self.common_transform is not None:
            both = torch.cat([input, target], dim=0)
            both = self.common_transform(both)
            (input, target) = torch.split(both, 3, dim=0)

        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return (input, target)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class ToDevice(torch.nn.Module):
    """
    Sends the input object to the device specified in the
    object's constructor by calling .to(device) on the object.
    """

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, img):
        return img.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"


def create_datasets(dataset_dir="dataset/", batch_size=16, image_size=128):
    pets_path_train = os.path.join(dataset_dir, "OxfordPets", "train")
    pets_path_test = os.path.join(dataset_dir, "OxfordPets", "test")

    def tensor_trimap(t):
        x = t * 255
        x = x.to(torch.long)
        x = x - 1
        return x

    def args_to_dict(**kwargs):
        return kwargs

    transform_dict = args_to_dict(
        pre_transform=T.ToTensor(),
        pre_target_transform=T.ToTensor(),
        common_transform=T.Compose(
            [
                ToDevice(get_device()),
                T.Resize(
                    (image_size, image_size), interpolation=T.InterpolationMode.NEAREST
                ),
                # Random Horizontal Flip as data augmentation.
                T.RandomHorizontalFlip(p=0.5),
            ]
        ),
        post_transform=T.Compose(
            [
                # Color Jitter as data augmentation.
                T.ColorJitter(contrast=0.3),
            ]
        ),
        post_target_transform=T.Compose(
            [
                T.Lambda(tensor_trimap),
            ]
        ),
    )

    # Create the train and test instances of the data loader for the
    # Oxford IIIT Pets dataset with random augmentations applied.
    # The images are resized to 128x128 squares, so the aspect ratio
    # will be chaged. We use the nearest neighbour resizing algorithm
    # to avoid disturbing the pixel values in the provided segmentation
    # mask.
    pets_train = OxfordIIITPetsAugmented(
        root=pets_path_train,
        split="trainval",
        target_types="segmentation",
        download=True,
        **transform_dict,
    )
    pets_test = OxfordIIITPetsAugmented(
        root=pets_path_test,
        split="test",
        target_types="segmentation",
        download=True,
        **transform_dict,
    )

    pets_train_loader = torch.utils.data.DataLoader(
        pets_train,
        batch_size=batch_size,
        shuffle=True,
    )
    pets_test_loader = torch.utils.data.DataLoader(
        pets_test,
        batch_size=batch_size,
        shuffle=True,
    )

    return pets_train_loader, pets_test_loader
