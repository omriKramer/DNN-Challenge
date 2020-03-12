from fastai.vision import *
import pre
import resample
from metrics import Pearson
import argparse

root = Path(__file__).resolve().parent / 'data'
train = root / 'train'
val = root / 'val'


def normalize_time(series):
    # 1440 minutes in a day
    normalized = (series.hour * 60 + series.minute) / 1440
    return normalized


def get_data(data_dir, sub_mean=False):
    cgm, meals = pre.get_dfs(data_dir)
    if sub_mean:
        mean, std = pre.norm_stats['GlucoseValue']
        cgm['GlucoseValue'] = cgm['GlucoseValue'] - mean / std

    meals = resample.resample_meals(cgm, meals, 15)
    meals = pd.concat((meals, cgm), axis=1)
    meals['time'] = normalize_time(meals.index.get_level_values('Date'))
    cgm, y = pre.build_cgm(cgm)
    return cgm, meals, y


class ContData(Dataset):
    def __init__(self, cgm, meals, y):
        self.cgm = cgm
        self.meals = meals
        self.y = y

    def __len__(self):
        return len(self.cgm)

    def __getitem__(self, i):
        index = self.meals.index.get_loc(self.cgm.index[i])
        values = self.meals[index - 48:index + 1].values
        target = self.y.iloc[i].values
        x, y = torch.tensor(values, dtype=torch.float), torch.tensor(target, dtype=torch.float)
        return x, y


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output[None], hidden)
        return output[0], hidden

    def initHidden(self, bs, device):
        return torch.zeros(1, bs, self.hidden_size, device=device)


MAX_LENGTH = 49


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights[:, None], encoder_outputs)

        output = torch.cat((embedded, attn_applied[:, 0]), 1)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output[None], hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights

    def initHidden(self, bs, device):
        return torch.zeros(1, bs, self.hidden_size, device=device)


class Seq2Seq(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size)

    def forward(self, input):
        device = input.device
        bs = input.shape[0]
        input = input.transpose(0, 1)

        encoder_hidden = self.encoder.initHidden(bs, device)
        encoder_outputs = input.new_zeros(bs, MAX_LENGTH, self.encoder.hidden_size)

        for ei in range(input.shape[0]):
            encoder_output, encoder_hidden = self.encoder(input[ei], encoder_hidden)
            encoder_outputs[:, ei] = encoder_output

        decoder_input = input.new_zeros(bs, 1)
        decoder_hidden = encoder_hidden

        out = []
        for di in range(8):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            out.append(decoder_output)
            decoder_input = decoder_output.detach()

        out = torch.cat(out, dim=1)
        return out


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=5, type=int)
parser.add_argument('--trainval', action='store_true')
args = parser.parse_args()

if args.trainval:
    train_data = get_data(root)
else:
    train_data = get_data(train)

val_data = get_data(val)

train_ds = ContData(*train_data)
val_ds = ContData(*val_data)
data = DataBunch.create(train_ds, val_ds, bs=512)

model = Seq2Seq(38, 128)
metrics = [mean_absolute_error, Pearson(val_ds.y)]
learner = Learner(data, model, loss_func=nn.MSELoss(), metrics=metrics)

if args.trainval:
    learner.fit_one_cycle(args.epochs, 1e-3)
    learner.save('gru-trainval')
else:
    save = callbacks.SaveModelCallback(learner, monitor='pearson')
    learner.fit_one_cycle(args.epochs, 1e-3, callbacks=save)
