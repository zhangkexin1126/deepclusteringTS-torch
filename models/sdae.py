'''
- PyTorch implementation of Denoising AutoEncoder:
- Stacked Denoising Autoencoders Learning Useful Representations in a
  Deep Network with a Local Denoising Criterion. JMLR2010
'''

from typing import Callable, Iterable, Optional, Tuple, List, Any
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from collections import OrderedDict
from cytoolz import itertoolz
from tqdm import tqdm
import utils
######################################################
def build_mlp_unit(
        dimensions: Iterable[int],
        activation: Optional[torch.nn.Module]
) -> List[torch.nn.Module]:
    """
    Build a list of units where each unit is a linear layer followed by an
    activation layer

    :param dimensions:
    :param activation:
    :return:
    """
    def single_unit(in_dimension: int, out_dimension: int) -> torch.nn.Module:
        unit = [("linear", nn.Linear(in_dimension, out_dimension))]
        if activation is not None:
            unit.append(("activation", activation))
        return nn.Sequential(OrderedDict(unit))

    return [
        single_unit(embedding_dimension, hidden_dimension)
        for embedding_dimension, hidden_dimension in itertoolz.sliding_window(2, dimensions)
    ]

def init_mlp_weight_bias_(
    weight: torch.Tensor, bias: torch.Tensor, gain: float
) -> None:
    torch.nn.init.xavier_uniform_(weight, gain)
    nn.init.constant_(bias, 0.0)

class denoising_mlp_ae(nn.Module):
    def __init__(
            self,
            dimensions: List[int],
            activation: torch.nn.Module = nn.ReLU(),
            final_activation: Optional[torch.nn.Module] = nn.ReLU(),
            weight_init: Callable[[torch.Tensor, torch.Tensor, float], None] = init_mlp_weight_bias_,
            gain: float = nn.init.calculate_gain("relu"),
    ):
        """

        """
        super(denoising_mlp_ae, self).__init__()
        self.dimensions = dimensions
        self.input_dimension = dimensions[0]
        self.hidden_dimension = dimensions[-1]
        # build encoder
        encoder_units = build_mlp_unit(self.dimensions[:-1], activation)
        encoder_units.extend(
            build_mlp_unit([self.dimensions[-2], self.dimensions[-1]], None)
        )
        self.encoder = nn.Sequential(*encoder_units)
        # build decoder
        decoder_units = build_mlp_unit(reversed(self.dimensions[1:]), activation)
        decoder_units.extend(
            build_mlp_unit([self.dimensions[1], self.dimensions[0]], final_activation)
        )
        self.decoder = nn.Sequential(*decoder_units)
        # Init the weights and biases
        for layer in itertoolz.concat([self.encoder, self.decoder]):
            weight_init(layer[0].weight, layer[0].bias, gain)

    def get_stack(self, index: int) -> Tuple[nn.Module, nn.Module]:
        '''

        :param index:
        :return:
        '''
        if (index >  len(self.dimensions) - 2 ) or (index < 0):
            raise ValueError(
                "Requested subautoencoder cannot be constructed, index out of range."
            )
        return self.encoder[index].linear, self.decoder[-(index + 1)].linear

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        z = self.encoder(batch)
        return z

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        z = self.encoder(batch)
        rec = self.decoder(z)
        return rec

class sub_mlpae_unit(nn.Module):
    def __init__(self,
                 first_dim: int,
                 second_dim: int,
                 activation: Optional[nn.Module] = nn.ReLU(),
                 gain: float = nn.init.calculate_gain('relu'),
                 corruption: Optional[nn.Module] = None,
                 tied: bool = False
                 ):
        super(sub_mlpae_unit, self).__init__()
        self.first_dim = first_dim
        self.second_dim = second_dim
        self.activation = activation
        self.gain = gain
        self.corruption = corruption
        # encoder parameters
        self.encoder_weight = nn.Parameter(torch.zeros((second_dim, first_dim)))
        self.encoder_bias = nn.Parameter(torch.zeros(second_dim))
        self._init_weight_bias(self.encoder_weight, self.encoder_bias, self.gain)
        # decoder parameters
        self._decoder_weight = (
            nn.Parameter(torch.zeros((first_dim, second_dim)))
            if not tied else None
        )
        self.decoder_bias = nn.Parameter(torch.zeros(first_dim))
        self._init_weight_bias(self.decoder_weight, self.decoder_bias, self.gain)

    @property
    def decoder_weight(self):
        return (self._decoder_weight if self._decoder_weight is not None
                else self.encoder_weight.t())

    @staticmethod
    def _init_weight_bias(weight: torch.Tensor, bias: torch.Tensor, gain: float):
        if weight is not None:
            nn.init.xavier_uniform_(weight, gain)
        nn.init.constant_(bias, 0)

    def copy_weight(self, encoder: torch.nn.Linear, decoder: torch.nn.Linear) -> None:
        encoder.weight.data.copy_(self.encoder_weight)
        encoder.bias.data.copy_(self.encoder_bias)
        decoder.weight.data.copy_(self.decoder_weight)
        decoder.bias.data.copy_(self.decoder_bias)
        print('-log info -> Finsh COPY')

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        #print('*********',self.encoder_weight.dtype, self.encoder_bias.dtype)
        h = F.linear(batch, self.encoder_weight, self.encoder_bias)
        #h = F.linear(batch, self.encoder_weight.double(), self.encoder_bias.double())
        if self.activation is not None:
            transformed = self.activation(h)
        if self.corruption is not None:
            transformed = self.corruption(h)
        return h

    def decode(self, batch: torch.Tensor) -> torch.Tensor:
        h = F.linear(batch, self.decoder_weight, self.decoder_bias)
        #h = F.linear(batch, self.decoder_weight.double(), self.decoder_bias.double())
        return h

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.decode(self.encode(batch))
        return out

def pretrain_mlp_dae(
        traindata: Dataset,
        autoencoder: denoising_mlp_ae,
        pretrain_epochs: int,
        batch_size: int,
        optimizer: Callable[[nn.Module], optim.Optimizer],
        scheduler: Optional[Callable[[optim.Optimizer], Any]] = None,
        validation: Optional[DataLoader] = None,
        corruption: Optional[float] = None,
        cuda: bool=False,

        silent: bool = False,
        update_freq: Optional[int] = 1
):
    current_dataset = traindata
    current_validation = validation
    num_subae = len(autoencoder.dimensions) - 1 # how many sub ae
    for idx in range(num_subae):
        subencoder, subdecoder = autoencoder.get_stack(idx)
        first_dim = autoencoder.dimensions[idx] # 784
        second_dim = autoencoder.dimensions[idx + 1] # 500
        print('-------------------------------------------------')
        print('-log info -> Training sub autoencoder: idx=', idx )
        print('-log info -> Sub-encoder dim:', first_dim, second_dim)
        print('-log info -> Sub-decoder dim:', second_dim, first_dim)
        # manual override to prevent corruption for the last subautoencoder
        if idx == (num_subae -1):
            corruption = None #last linear
        # Init subae
        print('-log info -> Start init weights and bias of subae:', idx)
        subae = sub_mlpae_unit(
            first_dim=first_dim,
            second_dim=second_dim,
            activation=nn.ReLU() if idx != (num_subae - 1) else None,
            corruption=nn.Dropout(corruption) if corruption is not None else None
        )
        print('Encoder:', subae.encoder_weight.shape, subae.encoder_bias.shape)
        #print(subae.encoder_weight.dtype, subae.encoder_bias.dtype)
        print('Decoder:', subae.decoder_weight.shape, subae.decoder_bias.shape)
        #print(subae.decoder_weight.dtype, subae.decoder_bias.dtype)
        print('-log info -> Finish init weights and bias of subae:', idx)
        if cuda:
            subae = subae.cuda()
        ae_optim = optimizer(subae)
        ae_scheduler = scheduler(ae_optim) if scheduler is not None else scheduler
        print('-log info -> Start training subae: idx=', idx)
        train_mlp_dae(
            current_dataset,
            subae,
            pretrain_epochs,
            batch_size,
            ae_optim,
            scheduler=ae_scheduler,
            validation=validation,
            corruption=None,
            cuda=cuda,
            silent=silent,
            update_freq=update_freq,
            loss_func = nn.MSELoss()
        )
        print('Finish Training subae: idx=', idx)
        # copy the weights
        subae.copy_weight(subencoder, subdecoder) # train a layer each time
        # pass
        print('Preparing training Dataloader for next subae')
        if idx != (num_subae - 1):
            print('idx', idx, 'num_subae', num_subae)
            current_dataset = predict_mlpdae(
                current_dataset, subae, batch_size=batch_size, cuda=cuda,silent=silent)[0]
            if current_validation is not None:
                current_validation = predict_mlpdae(
                    current_validation,subae,batch_size=batch_size,cuda=cuda,silent=silent)[0]
        else:
            current_dataset = None  # minor optimisation on the last subautoencoder
            current_validation = None

def train_mlp_dae(
        data: Dataset,
        autoencoder: nn.Module,
        train_epochs: int,
        batch_size: int,
        optimizer: optim.Optimizer,
        scheduler: Any = None,
        validation: Optional[DataLoader] = None,
        corruption: Optional[float] = None,
        cuda: bool = False,
        sampler: Optional[torch.utils.data.sampler.Sampler] = None,
        silent: bool = False,
        update_freq: Optional[int] = 1,

        loss_func: nn.Module = nn.MSELoss(),
):
    train_dl = utils.generate_dataloader(data,
                                         batch_size=batch_size,
                                         pin_memory=False,
                                         sampler=sampler,
                                         shuffle=True if sampler is None else False)

    if validation is not None:
        validation_dl = validation
    else:
        validation_dl = None
    autoencoder.double()
    autoencoder.train()
    validation_loss_value = -1
    loss_value = 0
    for epo in range(train_epochs):
        #print('-log info: epoch', epo)
        data_iterator = tqdm(
            train_dl,
            leave=True,
            unit="batch",
            postfix={"epo": epo, "training_loss": "%.6f" % 0.0, "validation_loss": "%.6f" % -1, },
            disable=silent)
        for idx, batch in enumerate(data_iterator):
            if (isinstance(batch, tuple) or isinstance(batch, list)
                    and len(batch) in [1, 2]):
                batch = batch[0]
            if cuda:
                batch = batch.cuda(non_blocking=True)
            batch = batch.view(batch.shape[0], -1)
            batch = batch.double()
            if corruption is not None:
                output = autoencoder(F.dropout(batch, corruption))
            else:
                output = autoencoder(batch)
            loss = loss_func(output, batch)
            # accuracy
            loss_value = float(loss.item())
            #print('--log info: loss_value', loss_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            data_iterator.set_postfix(
                epo=epo, training_loss="%.6f" % loss_value, validation_loss="%.6f" % validation_loss_value,
            )

        if scheduler is not None:
            scheduler.step()
        if update_freq is not None and epo % update_freq == 0:
            #print('-log info: Check Validation')
            if validation_dl is not None:
                validation_output = predict_mlpdae(validation_dl, autoencoder, cuda=cuda, silent=True,encode=False)[1]
                validation_inputs = []
                for val_batch in validation_dl:
                    if (
                            isinstance(val_batch, tuple) or isinstance(val_batch, list)
                    ) and len(val_batch) in [1, 2]:
                        validation_inputs.append(val_batch[0])
                    else:
                        validation_inputs.append(val_batch)
                validation_actual = torch.cat(validation_inputs)
                if cuda:
                    validation_actual = validation_actual.cuda(non_blocking=True)
                    validation_output = validation_output.cuda(non_blocking=True)
                validation_loss = loss_func(validation_output, validation_actual)
                validation_loss_value = float(validation_loss.item())
                print('-log info -> with validation, epoch=', epo)
                data_iterator.set_postfix(
                    epo=epo,
                    training_loss="%.6f" % loss_value,
                    validation_loss="%.6f" % validation_loss_value,
                )
                #autoencoder.train()
            else:
                validation_loss_value = -1
                #print('---log info: without validation, epoch=', epo)
                data_iterator.set_postfix(
                    epo=epo, training_loss="%.6f" % loss_value, validation_loss="%.6f" % -2,
                )

def predict_mlpdae(
        ds: Dataset,
        model: nn.Module,
        batch_size: int,
        cuda: bool = False,
        silent: bool = False,
        encode: bool = True
):

    dataloader = utils.generate_dataloader(ds,batch_size=batch_size, pin_memory=False, shuffle=False)
    data_iterator = tqdm(dataloader, leave=False, unit="batch", disable=silent)
    feats = []
    if isinstance(model, nn.Module):
        model.eval()
    for batch in data_iterator:
        if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
            batch = batch[0]
        if cuda:
            batch = batch.cuda(non_blocking=True)
        batch = batch.squeeze(1).view(batch.size(0), -1)
        if encode:
            output = model.encode(batch)
        else:
            output = model(batch)
        feats.append(output.detach().cpu())
        next_feat = torch.cat(feats)
    #print('&**&*&*&*&*&*&*&*feat_dim', next_feat.shape)
    next_td = TensorDataset(next_feat)
    if isinstance(model, nn.Module):
        model.train()
    return next_td, next_feat





