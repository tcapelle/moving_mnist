# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_seq2seq.ipynb (unless otherwise specified).

__all__ = ['ConvGRUCell', 'ConvGRU', 'dcgan_conv', 'dcgan_upconv', 'image_encoder', 'image_decoder', 'PhyConvGru',
           'Seq2Seq', 'Encoderloss', 'TeacherForcing']

# Cell
from fastai.vision.all import *
from fastai.text.models.awdlstm import RNNDropout
from .conv_rnn import TimeDistributed, StackUnstack, StackLoss
from .phy import *

# Cell
class ConvGRUCell(Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(3,3), bias=True, activation=F.tanh, batchnorm=False):
        """
        Initialize ConvGRU cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        self.input_dim          = input_dim
        self.hidden_dim         = hidden_dim

        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else [kernel_size]*2
        self.padding     = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        self.bias        = bias
        self.activation  = activation
        self.batchnorm   = batchnorm


        self.conv_zr = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=2 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.conv_h1 = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.conv_h2 = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.reset_parameters()

    def forward(self, input, h_prev=None):
        #init hidden on forward
        if h_prev is None:
            h_prev = self.init_hidden(input)

        combined = torch.cat((input, h_prev), dim=1)  # concatenate along channel axis

        combined_conv = F.sigmoid(self.conv_zr(combined))

        z, r = torch.split(combined_conv, self.hidden_dim, dim=1)

        h_ = self.activation(self.conv_h1(input) + r * self.conv_h2(h_prev))

        h_cur = (1 - z) * h_ + z * h_prev

        return h_cur

    def init_hidden(self, input):
        bs, ch, h, w = input.shape
        return one_param(self).new_zeros(bs, self.hidden_dim, h, w)

    def reset_parameters(self):
        #self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.conv_zr.weight, gain=nn.init.calculate_gain('tanh'))
        self.conv_zr.bias.data.zero_()
        nn.init.xavier_uniform_(self.conv_h1.weight, gain=nn.init.calculate_gain('tanh'))
        self.conv_h1.bias.data.zero_()
        nn.init.xavier_uniform_(self.conv_h2.weight, gain=nn.init.calculate_gain('tanh'))
        self.conv_h2.bias.data.zero_()

        if self.batchnorm:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()

# Cell
class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, n_layers, batch_first=True,
                 bias=True, activation=F.tanh, input_p=0.2, hidden_p=0.1, batchnorm=False):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, n_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, n_layers)
        activation  = self._extend_for_multilayer(activation, n_layers)

        if not len(kernel_size) == len(hidden_dim) == len(activation) == n_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bias = bias
        self.input_p = input_p
        self.hidden_p = hidden_p

        cell_list = []
        for i in range(self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvGRUCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          activation=activation[i],
                                          batchnorm=batchnorm))

        self.cell_list = nn.ModuleList(cell_list)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([nn.Dropout(hidden_p) for l in range(n_layers)])
        self.reset_parameters()

    def __repr__(self):
        s = f'ConvGru(in={self.input_dim}, out={self.hidden_dim[0]}, ks={self.kernel_size[0]}, '
        s += f'n_layers={self.n_layers}, input_p={self.input_p}, hidden_p={self.hidden_p})'
        return s
    def forward(self, input, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
        Returns
        -------
        last_state_list, layer_output
        """
        input = self.input_dp(input)
        cur_layer_input = torch.unbind(input, dim=int(self.batch_first))

        if hidden_state is None:
            hidden_state = self.get_init_states(cur_layer_input[0])

        seq_len = len(cur_layer_input)
        last_state_list   = []

        for l, (gru_cell, hid_dp) in enumerate(zip(self.cell_list, self.hidden_dps)):
            h = hidden_state[l]
            output_inner = []
            for t in range(seq_len):
                h = gru_cell(input=cur_layer_input[t], h_prev=h)
                output_inner.append(h)

            cur_layer_input = torch.stack(output_inner)  #list to array
            if l != self.n_layers: cur_layer_input = hid_dp(cur_layer_input)
            last_state_list.append(h)

        layer_output = torch.stack(output_inner, dim=int(self.batch_first))
        last_state_list = torch.stack(last_state_list, dim=0)
        return layer_output, last_state_list

    def reset_parameters(self):
        for c in self.cell_list:
            c.reset_parameters()

    def get_init_states(self, input):
        init_states = []
        for gru_cell in self.cell_list:
            init_states.append(gru_cell.init_hidden(input))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list)
            and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# Cell
class dcgan_conv(nn.Sequential):
    def __init__(self, nin, nout, stride):
        layers = [nn.Conv2d(nin, nout, kernel_size=(3,3), stride=stride, padding=1),
                  nn.GroupNorm(4,nout),
                  nn.LeakyReLU(0.2, inplace=True)]
        super().__init__(*layers)


class dcgan_upconv(nn.Sequential):
    def __init__(self, nin, nout, stride):
        layers = [nn.ConvTranspose2d(nin, nout,(3,3), stride=stride,
                                   padding=1,output_padding=1 if stride==2 else 0),
                  nn.GroupNorm(4,nout),
                  nn.LeakyReLU(0.2, inplace=True)]
        super().__init__(*layers)

# Cell
class image_encoder(Module):
    def __init__(self, nc=1):
        nf = 16
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, int(nf/2), stride=1) # (nf) x 64 x 64
        self.c2 = dcgan_conv(int(nf/2), nf, stride=1) # (nf) x 64 x 64
        self.c3 = dcgan_conv(nf, nf*2, stride=2) # (2*nf) x 32 x 32
        self.c4 = dcgan_conv(nf*2, nf*2, stride=1) # (2*nf) x 32 x 32
        self.c5 = dcgan_conv(nf*2, nf*4, stride=2) # (4*nf) x 16 x 16
        self.c6 = dcgan_conv(nf*4, nf*4, stride=1) # (4*nf) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)  # (nf/2) x 64 x 64
        h2 = self.c2(h1)     # (nf) x 64 x 64
        h3 = self.c3(h2)     # (2*nf) x 32 x 32
        h4 = self.c4(h3)     # (2*nf) x 32 x 32
        h5 = self.c5(h4)     # (4*nf) x 16 x 16
        h6 = self.c6(h5)     # (4*nf) x 16 x 16
        return h6 #[h1, h2, h3, h4, h5, h6]


class image_decoder(Module):
    def __init__(self, nc=1, nf=16):
        self.upc1 = dcgan_upconv(nf*4, nf*4, stride=1) #(nf*4) x 16 x 16
        self.upc2 = dcgan_upconv(nf*4, nf*2, stride=2) #(nf*2) x 32 x 32
        self.upc3 = dcgan_upconv(nf*2, nf*2, stride=1) #(nf*2) x 32 x 32
        self.upc4 = dcgan_upconv(nf*2, nf, stride=2)   #(nf) x 64 x 64
        self.upc5 = dcgan_upconv(nf, nf, stride=1)   #(nf/2) x 64 x 64
        self.upc6 = nn.ConvTranspose2d(in_channels=nf,out_channels=nc,kernel_size=(3,3),stride=1,padding=1)  #(nc) x 64 x 64

    def forward(self, vec):
        d1 = self.upc1(vec)  #(nf*4) x 16 x 16
        d2 = self.upc2(d1)   #(nf*2) x 32 x 32
        d3 = self.upc3(d2)   #(nf*2) x 32 x 32
        d4 = self.upc4(d3)   #(nf) x 64 x 64
        d5 = self.upc5(d4)   #(nf/2) x 64 x 64
        d6 = self.upc6(d5)   #(nc) x 64 x 64
        return d6

# Cell
class PhyConvGru(Module):
    def __init__(self, ks=3, n_layers=1, phy_ks=7, phy_n_layers=1):
        self.rnn = ConvGRU(64, 64, (ks, ks) if not isinstance(ks, tuple) else ks, n_layers)
        self.phy = PhyCell(64, [49], ks=phy_ks, n_layers=phy_n_layers)
    def forward(self, x, h=(None, None)):
        x, h_rnn = self.rnn(x, h[0])
        y, h_phy = self.phy(x, h[1])
        return x+y, (h_rnn, h_phy)

# Cell
class Seq2Seq(Module):
    "Simple seq2seq model"
    def __init__(self, seq_len=2, ch_out=1, ks=3, n_layers=1, use_phy=False, debug=False):
        store_attr()
        self.img_encoder = TimeDistributed(image_encoder())
        self.img_decoder = TimeDistributed(image_decoder(ch_out))
        self.rnn = ConvGRU(64, 64, (ks, ks) if not isinstance(ks, tuple) else ks, n_layers)
        self.phy = PhyCell(64, [49], ks=7, n_layers=1) if use_phy else None
        self.pr = 0.0

    def forward(self, x, targ=None):
        if self.debug: print('pr: ', self.pr)
        enc_imgs = self.img_encoder(x)
        if self.debug: print('enc_imgs shape', enc_imgs.shape)
        enc_outs, h = self.rnn(enc_imgs)
        dec_imgs = self.img_decoder(enc_outs)
        if self.use_phy:
            if self.debug: print('Computing PhyCell')
            phy_out, phy_h = self.phy(enc_imgs)
            if self.debug: print('phy_out.shape', phy_out.shape)
            dec_imgs = dec_imgs + self.img_decoder(phy_out)
        if targ is not None:
            if self.debug: print('targ is not None')
            enc_targs = self.img_encoder(targ) if targ is not None else None
            dec_inp = enc_targs[:, [0], ...]
            if self.use_phy: phy_inp = enc_targs[:, [0], ...]
        else:
            dec_inp = enc_outs[:, [-1], ...].detach()
            if self.use_phy: phy_inp = phy_out[:,[-1],...].detach()

        outs = [dec_imgs[:,[-1],...]]
        if self.debug: print('initial out:',  outs[0].shape)
        for i in range(self.seq_len-1):
            dec_inp, h = self.rnn(dec_inp, h)
            new_img = self.img_decoder(dec_inp)
            if self.use_phy:
                phy_inp, phy_h = self.phy(phy_inp, phy_h)
                new_img = new_img + self.img_decoder(phy_inp)
            outs.append(new_img)
            if self.debug: print('i out:',  outs[i].shape)
            if (targ is not None) and (random.random()<self.pr):
                if self.debug: print('pr i:', i)
                dec_inp = enc_targs[:,[i+1],:]
                if self.use_phy: phy_inp = enc_targs[:,[i+1],:]
        return torch.stack(outs, dim=1).squeeze(2), dec_imgs[:,:-1, ...], x[:,1:,...]

# Cell
class Encoderloss(Callback):
    def __init__(self, enc_loss_func, alpha=0.5):
        store_attr()
    def after_pred(self):
        self.learn.pred, self.learn.encoder_y, self.learn.encoder_targ = self.pred
    def after_loss(self):
        self.learn.loss += self.alpha*self.enc_loss_func(self.learn.encoder_y, self.learn.encoder_targ)

# Cell
class TeacherForcing(Callback):
    def __init__(self, end_epoch):
        self.end_epoch = end_epoch
    def before_batch(self):
        self.learn.xb = self.learn.xb + self.learn.yb
    def before_epoch(self):
        self.learn.model.module.pr = 1 - self.learn.epoch/self.end_epoch
    def before_validate(self):
        "force forecasting"
        self.learn.model.module.pr = 0