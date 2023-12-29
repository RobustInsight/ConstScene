from tqdm import tqdm
from torchvision import transforms
from tools.prepare_dataset import download_and_unzip
from constscene.segmentation_dataset_for_unet import get_loaders
from torch import nn, optim
import argparse
import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
import logging


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def calc_iou(output, target, num_classes):
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    output = torch.argmax(output, dim=1).to('cuda')
    target = target.squeeze(1).to('cuda')
    for cls in range(0, num_classes):
        intersection[cls] = torch.sum((output == cls) & (target == cls))
        union[cls] = torch.sum((output == cls) | (target == cls))

    return intersection, union


def check_accuracy(loader, model_, number_of_classes_, epoch_, device="cuda"):
    intersection = torch.zeros(number_of_classes_)
    union = torch.zeros(number_of_classes_)

    model_.eval()

    with torch.no_grad():
        for index, (img, mask) in tqdm(enumerate(loader), desc="Processing", total=len(loader), disable=True):
            img = img.to(device)
            preds = model_(img)

            intersect_, union_ = calc_iou(preds, mask, number_of_classes_)
            intersection += intersect_
            union += union_
            if index > 5:
                break

    miou_ = torch.nanmean(intersection / union)
    return miou_.float()


def train_fn(loader, model_, optimizer_, loss_fn_, device):
    loop_ = tqdm(loader, desc="          Training", disable=True)
    model_.train()

    for batch_idx, (image, mask) in enumerate(loop_):
        image = image.to(device)
        mask = mask.to(device)

        # forward
        predictions = model_(image)

        mask = mask.squeeze(1).type(torch.LongTensor).to(device)
        loss = loss_fn_(predictions, mask)

        # backward
        model_.zero_grad()
        loss.backward()
        optimizer_.step()


def Main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f'Process started')

    parser = argparse.ArgumentParser()

    # Command-line arguments
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of epochs to train the model')

    parser.add_argument('-d', '--database', default='D1', type=str, choices=('D1', 'D2'),
                        help='database- D1: data without augmentation, D2: data with augmentation')

    parser.add_argument('-lr', '--learning_rate', default=3e-4, type=float,
                        help='initial learning rate')

    parser.add_argument('-b', '--batch_size', default=1500, type=int,
                        help='batch size (default: 1500)')

    parser.add_argument('-ih', '--input_height', default=96, type=int,
                        help='input images height (default: 96)')

    parser.add_argument('-iw', '--input_width', default=128, type=int,
                        help='input images width (default: 28)')

    parser.add_argument('-en', '--encoder', default='resnet18', type=str, choices=('resnet18', 'resnet50'),
                        help='model encoder (read unet documents for more encoders. we have tested resnet18 and '
                             'resnet50)')

    parser.add_argument('--random_seed', default=420, type=int,
                        help='manual seed for random number generator')

    args = parser.parse_args()

    # Extracting values from command-line arguments
    epoch_numbers = args.epochs
    data_dir = '../data/' + args.database + '/'
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    encoder = args.encoder
    logger.info(f'Command-line arguments: {args}')

    seed_everything(args.random_seed)

    logger.info(f'Checking the input images. (downloading if needed)')
    download_and_unzip(args.database)

    number_of_classes = 7
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import time

    start_time = time.time()

    img_transform = transforms.Compose([
        transforms.Resize((args.input_height, args.input_width)),
        transforms.ToTensor()
    ])

    train_dir_ = os.path.join(data_dir, 'train/')
    val_dir_ = os.path.join(data_dir, 'valid/')
    test_dir_ = os.path.join(data_dir, 'test/')

    train_loader, val_loader, test_loader = get_loaders(train_dir_, val_dir_, test_dir_,
                                                        batch_size, img_transform, img_transform, img_transform)

    model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                     in_channels=3, classes=number_of_classes).to(device)

    print('Model Size : ', sum(p.numel() for p in model.parameters()))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    save_id = str(int(time.time()))

    loop = tqdm(range(epoch_numbers), desc="Epochs")
    for epoch, _ in enumerate(loop):

        # train
        train_fn(train_loader, model, optimizer, loss_fn, device)

        # check accuracy
        miou = check_accuracy(val_loader, model, number_of_classes, epoch, device=device)
        if epoch % 9 == 0:
            checkpoint_path = '../saved_checkpoints/unet_' + args.encoder + '_' + args.database + '_' + save_id
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path + '/unet_' + args.encoder + '_' + args.database+'_' + str(epoch) + '.ckpt')
        loop.set_postfix_str(f"Epoch: {epoch}/{epoch_numbers}, mIoU: {miou * 100:.4f}")

    # test
    miou = check_accuracy(test_loader, model, number_of_classes, epoch, device=device)
    print('test mIoU :', miou * 100)
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total running time: {total_time:.0f} seconds")
    logger.info(f'Done !')


if __name__ == '__main__':
    Main()
