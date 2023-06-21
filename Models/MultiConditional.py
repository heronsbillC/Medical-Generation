import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import Models.Conditional as Conditional
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence
from Models.models import PositionalEncoding

from Models.SubLayers import MultiHeadAttention


DROPOUT = 0.1  # Avoid overfitting
NUM_EMBEDS = 256
FWD_DIM = 512

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


class ConditionText(nn.Module):
    """conT
    """

    def __init__(self, d_model, n_head=8, d_ff=2048, dropout=0.1, N=1, k=49):
        super(ConditionText, self).__init__()
        self.N = N
        self.attI = MultiHeadAttention(n_head, d_model, dropout)
        self.attT = MultiHeadAttention(n_head, d_model, dropout)
        self.w1 = nn.Linear(d_model, 1)
        self.w2 = nn.Linear(d_model, 1)
        self.gates = None

    def forward(self, hiddents, V, T):
        CV, SV = torch.mean(V[:, -1], dim=1, keepdim=True), torch.mean(V[:, :-1], dim=2)
        scores = self.w1(CV.expand_as(SV)) + self.w2(SV)
        # gate = F.softmax(scores.squeeze(-1), dim=-1)
        gate = scores.squeeze(-1)  # [bs, N]
        # import ipdb
        # ipdb.set_trace()
        if self.gates == None:
            self.gates = gate
        else:
            self.gates = torch.cat((self.gates, gate), 0)

        Vs, Ts = [], []
        curr_vf = V[:, -1]
        for i in range(self.N):
            prev_vf = V[:, i]
            prev_tf = T[:, i]

            delta_vf = curr_vf - prev_vf  # bs x 49 x d_model
            #  bs x 49 x d_model
            Vs.append(delta_vf)
            #  Weighted textual features
            Ts.append(gate[:, i][:, None, None] * prev_tf)
        # bs x M x 49 x d_model -> bs x M*49 x d_model
        DeltaV = torch.cat(Vs, dim=1)
        T_ = torch.cat(Ts, dim=1)
        conI = self.attI(hiddents, DeltaV, DeltaV)
        conT = self.attT(hiddents, T_, T_)

        return conI, conT


# Caption Decoder
class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, N=1, v_size=49,
                 num_layers=6, d_model=256, nhead=8, dim_feedforward=FWD_DIM, dropout=DROPOUT):
        super(Decoder, self).__init__()
        self.N = N
        self.nhead = nhead

        # word embedding
        self.caption_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,dropout=dropout),
            num_layers=4)

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers
        )
        # Conditional Report
        self.condition = ConditionText(embed_size, n_head=nhead, d_ff=dim_feedforward, N=N)
        # Attention Block
        self.attention = MultiHeadAttention(heads=nhead, d_model=hidden_size, dropout=dropout)
        # Final Caption generator
        self.mlp = nn.Linear(hidden_size * 4, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.mlp.weight, mode='fan_in')
        self.mlp.bias.data.fill_(0)

    # images, prev_repo, target
    def forward(self, V, T, captions, basic_model, states=None):
        # Word Embedding, bs x len x d_model
        embeddings = self.caption_embed(captions)
        embeddings = embeddings.transpose(0, 1)  # seq_len x bs x d_model
        embeddings = self.pos_encoder(embeddings)  # seq_len x bs x d_model
        # Encoder
        # encoded = self.transformer_encoder(embeddings)

        # bs x 49 x d_model
        _, curr_vf = V[:, 0], V[:, -1]

        # Transformer Decoder Block
        tgt = embeddings
        memory = curr_vf.permute(1, 0, 2)
        # Transformer解码器模块中，如果没有提供掩码，则默认使用全1矩阵作为掩码，表示模型可以使用所有的目标序列信息进行预测
        # 用于屏蔽目标序列中当前时间步之后的位置，以避免模型在预测时使用未来的信息
        # 产生一个上三角矩阵，下三角的值全为1，上三角的值权威0，对角线也是1。
        tgt_mask = nn.Transformer().generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        tgt_mask = tgt_mask.unsqueeze(0).expand(tgt.size(1) * self.nhead, -1,
                                                -1)  # tgt_mask的形状应该是(seq_len, seq_len)，其中seq_len是目标序列的长度
        memory_mask = None
        # tgt：解码器模块的目标序列，目标序列通常是指图像描述中的文字序列 memory：编码器模块的输出序列，图像特征
        # 目标序列通常是指图像描述中的文本序列。在图像描述任务中，我们希望生成一个与输入图片相对应的文字序列，因此我们需要将输入图片的视觉信息融入到生成的文本序列中，以保证生成的描述与输入图片相符。
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        conI, conT = self.condition(output.permute(1, 0, 2), V, T)
        ctx = self.attention(output.permute(1, 0, 2), curr_vf, curr_vf)
        output = torch.cat([output.permute(1, 0, 2), ctx, conI, conT], dim=2)
        # Final score along vocabulary
        #  bs x len x vocab_size
        scores = self.mlp(self.dropout(output))

        # Return states for Caption Sampling purpose
        return scores


# Whole Architecture with Image Encoder and Caption decoder
class Encoder2Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, N):
        super(Encoder2Decoder, self).__init__()

        # Image CNN encoder
        self.encoder_image = Conditional.EncoderCNN(embed_size, hidden_size, N)

        # Concept encoder
        self.encoder_concept = Conditional.EncoderTXT(vocab_size, embed_size, hidden_size, N)

        # Caption Decoder
        self.decoder = Decoder(embed_size, vocab_size, hidden_size, N)

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

