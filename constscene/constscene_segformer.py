import argparse

import torch.utils.collect_env
import torch
import time
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor

from tools.prepare_dataset import download_and_unzip
from tools.seg_tools import write_text_in_file
from constscene.segmentation_dataset_for_segformer import SemanticSegmentationDataset, SegformerFinetuner


def Main():
    torch.set_float32_matmul_precision('medium')

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f'Process started')

    parser = argparse.ArgumentParser()

    # Command-line arguments
    parser.add_argument('--epochs', default=64, type=int,
                        help='number of epochs to train the model')

    parser.add_argument('--model', default='b0', type=str, choices=('b0', 'b0'),
                        help='segformer model (b0 or b5)')

    parser.add_argument('-d', '--database', default='D1', type=str, choices=('D1', 'D2'),
                        help='database- D1: data without augmentation, D2: data with augmentation')

    parser.add_argument('-lr', '--learning_rate', default=3e-4, type=float,
                        help='initial learning rate')

    parser.add_argument('-b', '--batch_size', default=300, type=int,
                        help='batch size (default: 1500)')

    parser.add_argument('-ih', '--input_height', default=96, type=int,
                        help='input images height (default: 96)')

    parser.add_argument('-iw', '--input_width', default=128, type=int,
                        help='input images width (default: 28)')

    parser.add_argument('--feature_size', default=128, type=int,
                        help='feature extractor size')

    args = parser.parse_args()



    # Extracting values from command-line arguments
    epoch_numbers = args.epochs
    data_dir = '../data/' + args.database + '/'
    segformer_model = args.model
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    logger.info(f'Command-line arguments: {args}')

    if segformer_model == 'b0':
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    elif segformer_model == 'b5':
        model_name = "nvidia/segformer-b5-finetuned-ade-640-640"


    logger.info(f'Checking the input images. (downloading if needed)')
    download_and_unzip(args.database)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print('Starting .... ')

    start_time = time.time()

    logging.getLogger("pytorch_lightning.utilities.rank_zero").addHandler(logging.NullHandler())
    logging.getLogger("pytorch_lightning.accelerators.cuda").addHandler(logging.NullHandler())

    feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
    feature_extractor.do_reduce_labels = False
    feature_extractor.size = args.feature_size

    train_dataset = SemanticSegmentationDataset("./" + data_dir + "/train/", feature_extractor)
    val_dataset = SemanticSegmentationDataset("./" + data_dir + "/valid/", feature_extractor)
    test_dataset = SemanticSegmentationDataset("./" + data_dir + "/test/", feature_extractor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, prefetch_factor=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=3, prefetch_factor=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=3, prefetch_factor=8)

    segformer_finetuner = SegformerFinetuner(
        train_dataset.id2label,
        model_name,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        metrics_interval=10,
    )

    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="mean_iou",
        mode="max"

    )

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=epoch_numbers,
        val_check_interval=len(train_dataloader),
        callbacks=[checkpoint_callback]

    )
    print('fitting srated....')

    trainer.fit(segformer_finetuner)
    print('fitting finished. testing started....')

    res = trainer.test(ckpt_path="last")


    print(res)

    end_time = time.time()
    total_time = end_time - start_time

    logger.info(f"Total running time: {total_time:.0f} seconds")
    logger.info(f"Done !")


if __name__ == '__main__':
    Main()
