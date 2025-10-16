import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3, bias=True):
        super().__init__()
        p = k // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=p, bias=bias)

    def forward(self, x, h, c):
        if h is None:
            h = torch.zeros(x.size(0), self.hid_ch, x.size(2), x.size(3),
                            device=x.device, dtype=x.dtype)
            c = torch.zeros_like(h)
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(z, 4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batch'):
        super().__init__()
        if norm == 'batch':
            Norm = lambda ch: nn.BatchNorm2d(ch)
        elif norm == 'inst':
            Norm = lambda ch: nn.InstanceNorm2d(ch, affine=False)
        else:
            Norm = lambda ch: nn.Identity()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            Norm(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            Norm(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, base=32, norm='batch'):
        super().__init__()
        self.e1 = ConvBlock(1, base, norm)
        self.e2 = ConvBlock(base, base*2, norm)
        self.e3 = ConvBlock(base*2, base*4, norm)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        return x3, x2, x1

class Decoder(nn.Module):
    def __init__(self, base=32, skip_dropout=0.0):
        super().__init__()
        self.drop2 = nn.Dropout2d(skip_dropout) if skip_dropout > 0 else nn.Identity()
        self.drop1 = nn.Dropout2d(skip_dropout) if skip_dropout > 0 else nn.Identity()
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d1 = ConvBlock(base*4, base*2)
        self.up2 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d2 = ConvBlock(base*2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, b3, s2, s1):
        y = self.up1(b3)
        y = torch.cat([y, self.drop2(s2)], dim=1)
        y = self.d1(y)
        y = self.up2(y)
        y = torch.cat([y, self.drop1(s1)], dim=1)
        y = self.d2(y)
        return self.head(y)

class ConvLSTM_UNet_Predictor(nn.Module):
    def __init__(self, base=32, skip_dropout=0.0, norm='batch'):
        super().__init__()
        self.enc = Encoder(base, norm)
        self.dec = Decoder(base, skip_dropout)
        self.lstm_b3 = ConvLSTMCell(base*4, base*4, 3)
        self.lstm_s2 = ConvLSTMCell(base*2, base*2, 3)
        self.lstm_s1 = ConvLSTMCell(base, base, 3)

    def forward(self, x_seq):
        if isinstance(x_seq, (tuple, list)):
            x_seq = x_seq[0]
        B, T, C, H, W = x_seq.shape
        hb = cb = hs2 = cs2 = hs1 = cs1 = None
        last_b3 = last_s2 = last_s1 = None

        for t in range(T):
            xt = x_seq[:, t]
            b3, s2, s1 = self.enc(xt)
            last_b3, last_s2, last_s1 = b3, s2, s1
            hb, cb = self.lstm_b3(b3, hb, cb)
            hs2, cs2 = self.lstm_s2(s2, hs2, cs2)
            hs1, cs1 = self.lstm_s1(s1, hs1, cs1)

        pred_logits = self.dec(hb, hs2, hs1)
        recon_logits = self.dec(last_b3, last_s2, last_s1)
        return pred_logits, recon_logits

if __name__ == '__main__':
    model = ConvLSTM_UNet_Predictor(base=16, skip_dropout=0.0, norm='batch')
    x = torch.randn(2, 5, 1, 128, 128)
    pred, recon = model(x)
    print('Prediction shape:', pred.shape)
    print('Reconstruction shape:', recon.shape)
