import copy
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence

from Models.SubLayers import MultiHeadAttention


def init_weight(f):
    init.kaiming_uniform_(f.weight, mode='fan_in')
    f.bias.data.fill_(0)
    return f


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class EncoderCNN(nn.Module):
    """Encoder image"""

    def __init__(self, embed_size, hidden_size, N):
        super(EncoderCNN, self).__init__()
        self.N = N
        self.d_model = hidden_size

        # ResNet-152 backend
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential(*modules)  # last conv feature

        self.resnet_conv = resnet_conv
        self.avgpool = nn.AvgPool2d(7)
        self.affine_as = get_clones(nn.Linear(2048, hidden_size), N + 1)

        # Dropout before affine transformation
        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        for i in range(self.N + 1):
            self.affine_as[i] = init_weight(self.affine_as[i])

    def forward(self, images):
        '''
        Input: images
        Output: V=[v_1, ..., v_n]
        '''
        bs, n, c, h, w = images.size()
        V = torch.zeros((bs, self.N + 1, 49, self.d_model))
        if torch.cuda.is_available():
            V = V.cuda()

        for i in range(self.N + 1):
            # Last conv layer feature map
            # previous image feature: bs x feature_size x 7 x 7
            A = self.resnet_conv(images[:, i])
            # bs x 49 x feature_size
            feat = A.view(bs, A.size(1), -1).transpose(1, 2)
            # bs x 49 x d_model
            feat = F.relu(self.affine_as[i](self.dropout(feat)))
            # bs x (N+1) x 49 x d_model
            V[:, i] = feat

        return V


class EncoderTXT(nn.Module):
    """encode conditional report"""

    def __init__(self, vocab_size, embed_size, hidden_size, N):
        super(EncoderTXT, self).__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.BiLSTM = nn.LSTM(embed_size, hidden_size // 2, num_layers=2, bidirectional=True)

    def forward(self, input):
        # encode
        # embedding: bs x N x tlen -> bs x N x tlen x d_model
        embed_report = self.embed(input)

        bs, N, tlen, d_model = embed_report.size()
        align_report = torch.zeros((bs, self.N, tlen, d_model))
        if torch.cuda.is_available():
            align_report = align_report.cuda()

        for i in range(self.N):
            #  bs x tlen x d_model
            align_report_, _ = self.BiLSTM(embed_report[:, i])
            align_report[:, i] = align_report_

        return align_report


# Caption Decoder
class DecoderIcRc(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, N=1, v_size=49):
        super(DecoderIcRc, self).__init__()
        self.N = N

        self.affine_va = nn.Linear(hidden_size, embed_size)

        # word embedding
        self.caption_embed = nn.Embedding(vocab_size, embed_size)

        # LSTM decoder
        self.LSTM = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)

        # Save hidden_size for hidden variable
        self.hidden_size = hidden_size

        # Attention Block
        self.attention = MultiHeadAttention(heads=8, d_model=hidden_size)

        # Final Caption generator
        self.mlp = nn.Linear(hidden_size * 4, vocab_size)

        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        init.kaiming_uniform_(self.affine_va.weight, mode='fan_in')
        self.affine_va.bias.data.fill_(0)
        init.kaiming_normal_(self.mlp.weight, mode='fan_in')
        self.mlp.bias.data.fill_(0)

    def forward(self, V, T, captions, basic_model, states=None):
        # Word Embedding, bs x len x d_model
        embeddings = self.caption_embed(captions)
        bs, tlen, d_m = embeddings.size()

        # Hiddens: Batch x seq_len x hidden_size
        hiddents = torch.zeros((bs, tlen, d_m))
        if torch.cuda.is_available():
            hiddents = hiddents.cuda()

        # bs x 49 x d_model
        prev_vf, curr_vf = V[:, 0], V[:, -1]
        # bs x len x d_model
        x = embeddings
        # Recurrent Block
        for time_step in range(embeddings.size(1)):
            # Feed in x_t one at a time
            x_t = x[:, time_step, :]  # bs x d_model
            x_t = x_t.unsqueeze(1)  # bs x 1 x d_model

            # h_t: # bs x 1 x d_model
            h_t, states = self.LSTM(x_t, states)  # Batch_first

            # Save hidden
            hiddents[:, time_step] = h_t.squeeze(1)
        # ipdb.set_trace()
        prev_vf = torch.mean(prev_vf, dim=1, keepdim=True).expand_as(hiddents)
        prev_tf = torch.mean(T[:,0 ], dim=1, keepdim=True).expand_as(hiddents)
        ctx = self.attention(hiddents, curr_vf, curr_vf)
        output = torch.cat([hiddents, ctx, prev_vf, prev_tf], dim=2)
        # Final score along vocabulary
        #  bs x len x vocab_size
        scores = self.mlp(self.dropout(output))

        # Return states for Caption Sampling purpose
        return scores


# Caption Decoder
class DecoderIc(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, N=1, v_size=49):
        super(DecoderIc, self).__init__()
        self.N = N

        self.affine_va = nn.Linear(hidden_size, embed_size)

        # word embedding
        self.caption_embed = nn.Embedding(vocab_size, embed_size)

        # LSTM decoder
        self.LSTM = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)

        # Save hidden_size for hidden variable
        self.hidden_size = hidden_size

        # Attention Block
        self.attention = MultiHeadAttention(heads=8, d_model=hidden_size)

        # Final Caption generator
        self.mlp = nn.Linear(hidden_size * 3, vocab_size)

        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        init.kaiming_uniform_(self.affine_va.weight, mode='fan_in')
        self.affine_va.bias.data.fill_(0)
        init.kaiming_normal_(self.mlp.weight, mode='fan_in')
        self.mlp.bias.data.fill_(0)

    def forward(self, V, T, captions, basic_model, states=None):
        # Word Embedding, bs x len x d_model
        embeddings = self.caption_embed(captions)
        bs, tlen, d_m = embeddings.size()

        # Hiddens: Batch x seq_len x hidden_size
        hiddents = torch.zeros((bs, tlen, d_m))
        if torch.cuda.is_available():
            hiddents = hiddents.cuda()

        # bs x 49 x d_model
        prev_vf, curr_vf = V[:, 0], V[:, -1]
        # bs x len x d_model
        x = embeddings
        # Recurrent Block
        for time_step in range(embeddings.size(1)):
            # Feed in x_t one at a time
            x_t = x[:, time_step, :]  # bs x d_model
            x_t = x_t.unsqueeze(1)  # bs x 1 x d_model

            # h_t: # bs x 1 x d_model
            h_t, states = self.LSTM(x_t, states)  # Batch_first

            # Save hidden
            hiddents[:, time_step] = h_t.squeeze(1)
        # ipdb.set_trace()
        prev_vf = torch.mean(prev_vf, dim=1, keepdim=True).expand_as(hiddents)
        ctx = self.attention(hiddents, curr_vf, curr_vf)
        output = torch.cat([hiddents, ctx, prev_vf], dim=2)
        # Final score along vocabulary
        #  bs x len x vocab_size
        scores = self.mlp(self.dropout(output))

        # Return states for Caption Sampling purpose
        return scores


# Caption Decoder
class DecoderRc(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, N=1, v_size=49):
        super(DecoderRc, self).__init__()
        self.N = N

        self.affine_va = nn.Linear(hidden_size, embed_size)

        # word embedding
        self.caption_embed = nn.Embedding(vocab_size, embed_size)

        # LSTM decoder
        self.LSTM = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)

        # Save hidden_size for hidden variable
        self.hidden_size = hidden_size

        # Attention Block
        self.attention = MultiHeadAttention(heads=8, d_model=hidden_size)

        # Final Caption generator
        self.mlp = nn.Linear(hidden_size * 3, vocab_size)

        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        init.kaiming_uniform_(self.affine_va.weight, mode='fan_in')
        self.affine_va.bias.data.fill_(0)
        init.kaiming_normal_(self.mlp.weight, mode='fan_in')
        self.mlp.bias.data.fill_(0)

    def forward(self, V, T, captions, basic_model, states=None):
        # Word Embedding, bs x len x d_model
        embeddings = self.caption_embed(captions)
        bs, tlen, d_m = embeddings.size()

        # Hiddens: Batch x seq_len x hidden_size
        hiddents = torch.zeros((bs, tlen, d_m))
        if torch.cuda.is_available():
            hiddents = hiddents.cuda()

        # bs x 49 x d_model
        prev_vf, curr_vf = V[:, 0], V[:, -1]
        # bs x len x d_model
        x = embeddings
        # Recurrent Block
        for time_step in range(embeddings.size(1)):
            # Feed in x_t one at a time
            x_t = x[:, time_step, :]  # bs x d_model
            x_t = x_t.unsqueeze(1)  # bs x 1 x d_model

            # h_t: # bs x 1 x d_model
            h_t, states = self.LSTM(x_t, states)  # Batch_first

            # Save hidden
            hiddents[:, time_step] = h_t.squeeze(1)

        prev_tf = torch.mean(T[:, 0], dim=1, keepdim=True).expand_as(hiddents)
        ctx = self.attention(hiddents, curr_vf, curr_vf)
        output = torch.cat([hiddents, ctx, prev_tf], dim=2)
        # Final score along vocabulary
        #  bs x len x vocab_size
        scores = self.mlp(self.dropout(output))

        # Return states for Caption Sampling purpose
        return scores

# Whole Architecture with Image Encoder and Caption decoder
class Encoder2DecoderIcRc(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, N):
        super(Encoder2DecoderIcRc, self).__init__()

        # Image CNN encoder
        self.encoder_image = EncoderCNN(embed_size, hidden_size, N)

        # Concept encoder
        self.encoder_concept = EncoderTXT(vocab_size, embed_size, hidden_size, N)

        # Caption Decoder
        self.decoder = DecoderIcRc(embed_size, vocab_size, hidden_size, N)

        # Share the weight matrix between caption & concept word embeddings
        self.encoder_concept.embed.weight = self.decoder.caption_embed.weight

        assert embed_size == hidden_size, "The values of embed_size and hidden_size should be equal."

    def forward(self, images, captions, image_concepts, lengths, basic_model):
        # imag -> V : bs x C x H x W -> bs x 2 x 49 x d_model
        # concept -> T: bs x tlen -> bs x tlen x d_model
        V = self.encoder_image(images)
        T = self.encoder_concept(image_concepts)

        # Language Modeling on word prediction
        # bs x len x vocab_size
        scores = self.decoder(V, T, captions, basic_model)

        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence(scores, lengths, batch_first=True)

        return packed_scores


# Whole Architecture with Image Encoder and Caption decoder
class Encoder2DecoderIc(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, N):
        super(Encoder2DecoderIc, self).__init__()

        # Image CNN encoder
        self.encoder_image = EncoderCNN(embed_size, hidden_size, N)

        # Concept encoder
        self.encoder_concept = EncoderTXT(vocab_size, embed_size, hidden_size, N)

        # Caption Decoder
        self.decoder = DecoderIc(embed_size, vocab_size, hidden_size, N)

        # Share the weight matrix between caption & concept word embeddings
        self.encoder_concept.embed.weight = self.decoder.caption_embed.weight

        assert embed_size == hidden_size, "The values of embed_size and hidden_size should be equal."

    def forward(self, images, captions, image_concepts, lengths, basic_model):
        # imag -> V : bs x C x H x W -> bs x 2 x 49 x d_model
        # concept -> T: bs x tlen -> bs x tlen x d_model
        V = self.encoder_image(images)
        T = self.encoder_concept(image_concepts)

        # Language Modeling on word prediction
        # bs x len x vocab_size
        scores = self.decoder(V, T, captions, basic_model)

        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence(scores, lengths, batch_first=True)

        return packed_scores


# Whole Architecture with Image Encoder and Caption decoder
class Encoder2DecoderRc(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, N):
        super(Encoder2DecoderRc, self).__init__()

        # Image CNN encoder
        self.encoder_image = EncoderCNN(embed_size, hidden_size, N)

        # Concept encoder
        self.encoder_concept = EncoderTXT(vocab_size, embed_size, hidden_size, N)

        # Caption Decoder
        self.decoder = DecoderRc(embed_size, vocab_size, hidden_size, N)

        # Share the weight matrix between caption & concept word embeddings
        self.encoder_concept.embed.weight = self.decoder.caption_embed.weight

        assert embed_size == hidden_size, "The values of embed_size and hidden_size should be equal."

    def forward(self, images, captions, image_concepts, lengths, basic_model):
        # imag -> V : bs x C x H x W -> bs x 2 x 49 x d_model
        # concept -> T: bs x tlen -> bs x tlen x d_model
        V = self.encoder_image(images)
        T = self.encoder_concept(image_concepts)

        # Language Modeling on word prediction
        # bs x len x vocab_size
        scores = self.decoder(V, T, captions, basic_model)

        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence(scores, lengths, batch_first=True)

        return packed_scores
