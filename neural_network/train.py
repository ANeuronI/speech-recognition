import os
import ast
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from model import SpeechRecognition
from dataset import Data, collate_fn_padd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpeechModule(LightningModule):

    def __init__(self, model, args):
        super(SpeechModule, self).__init__()
        self.model = model
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
        self.args = args
        self.val_epoch_outputs = []  # Initialize val_epoch_outputs

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.50, patience=6, min_lr=1e-6,
            verbose=True
        )
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler,
            'monitor': 'val_loss' 
        }   

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch 
        if spectrograms is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)  # Return a dummy loss if spectrograms is None
        bs = spectrograms.shape[0]
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)

        output, _ = self(spectrograms, (hn, c0))
        output = F.log_softmax(output, dim=2)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        logs = {'loss': loss, 'lr': self.optimizer.param_groups[0]['lr'] }
        return {'loss': loss, 'log': logs}

    def train_dataloader(self):
        d_params = Data.parameters
        d_params.update(self.args.dparams_override)
        train_dataset = Data(json_path=self.args.train_file, **d_params)
        return DataLoader(dataset=train_dataset,
                            batch_size=self.args.batch_size,
                            num_workers=15,
                            pin_memory=True,
                            collate_fn=collate_fn_padd)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.val_epoch_outputs.append({'val_loss': loss})  # Append loss to val_epoch_outputs
        return loss

    def on_validation_epoch_end(self):
        if not self.val_epoch_outputs:
            print("No validation outputs collected.")
            return

        avg_loss = torch.stack([x['val_loss'] for x in self.val_epoch_outputs]).mean()
        self.log('val_loss', avg_loss)
        self.scheduler.step(avg_loss)

    def val_dataloader(self):
        d_params = Data.parameters
        d_params.update(self.args.dparams_override)
        test_dataset = Data(json_path=self.args.valid_file, **d_params, valid=True)
        return DataLoader(dataset=test_dataset,
                            batch_size=self.args.batch_size,
                            num_workers=15,
                            collate_fn=collate_fn_padd,
                            pin_memory=True)


def main(args):
    h_params = SpeechRecognition.hyper_parameters
    h_params.update(args.hparams_override)
    model = SpeechRecognition(**h_params)

    if args.load_model_from:
        speech_module = SpeechModule.load_from_checkpoint(args.load_model_from, model=model, args=args)
    else:
        speech_module = SpeechModule(model, args)

    logger = TensorBoardLogger(args.logdir, name='speech_recognition')
    trainer = Trainer(logger=logger, max_epochs=args.epochs, val_check_interval=15)
    trainer.fit(speech_module)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train_file', default=None, required=True, type=str,
                        help='json file to load training data')
    parser.add_argument('--valid_file', default=None, required=True, type=str,
                        help='json file to load testing data')
    parser.add_argument('--valid_every', default=1000, required=False, type=int,
                        help='validate after every N iteration')
    parser.add_argument('--save_model_path', default=None, required=True, type=str,
                        help='path to save model')
    parser.add_argument('--load_model_from', default=None, required=False, type=str,
                        help='path to load a pretrain model to continue training')
    parser.add_argument('--logdir', default='tb_logs', required=False, type=str,
                        help='path to save logs')
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument("--hparams_override", default="{}", type=str, required=False,
                        help='override the hyper parameters, should be in form of dict. ie. {"attention_layers": 16 }')
    parser.add_argument("--dparams_override", default="{}", type=str, required=False,
                        help='override the data parameters, should be in form of dict. ie. {"sample_rate": 8000 }')
    args = parser.parse_args()
    args.hparams_override = ast.literal_eval(args.hparams_override)
    args.dparams_override = ast.literal_eval(args.dparams_override)

    if args.save_model_path:
        if not os.path.isdir(os.path.dirname(args.save_model_path)):
            raise Exception("the directory for path {} does not exist".format(args.save_model_path))

    main(args)
