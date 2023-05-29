import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence
from Models.models import TransformerLayer

from Models.SubLayers import MultiHeadAttention

DROPOUT = 0.1  # Avoid overfitting
NUM_HEADS = 8
NUM_LAYERS = 1

def init_weight(f):
    init.kaiming_uniform_(f.weight, mode='fan_in')
    f.bias.data.fill_(0)
    return f


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class EncoderCNN(nn.Module):
    """Encoder image"""
    # 创建对象时需要初始化的参数
    def __init__(self,embed_size, hidden_size, N):
        # N是相似图像的个数，此处为1
        super(EncoderCNN, self).__init__()
        self.N = N
        self.d_model = hidden_size

        # model =  models.resnet50(pretrained=True)
        model = models.densenet121(pretrained=True)
        modules = list(model.children())[:-1]  # delete the last fc layer and avg pool.
        # # 删除了模型的最后一个全连接层和平均池化层，得到一个只包含卷积层的特征提取器
        self.feature = nn.Sequential(*modules) # last conv feature
        self.avgpool = nn.AvgPool2d(7) #用于将卷积层的输出特征做平均池化，从而生成更紧凑的特征表示。
        self.affine_as = get_clones(nn.Linear(1024, hidden_size), N + 1) #定义全连接层，用于将池化后的特征映射到指定的隐藏层维度hidden_size。

        # Dropout before affine transformation
        self.dropout = nn.Dropout(0.5) #用于在全连接层之前对特征进行随机失活，以减轻过拟合。
        self.init_weights() #初始化全连接层的权重

    def init_weights(self):
        """Initialize the weights."""
        for i in range(self.N + 1):
            self.affine_as[i] = init_weight(self.affine_as[i])
    # 在调用时需要传入的参数
    def forward(self, images):
        '''
        Input: images
        Output: V=[v_1, ..., v_n]
        '''
        '''
        bs：批处理大小（即一次处理的图像数量）
        c：图像通道数（例如，灰度图像为1，RGB图像为3）
        n：图像序列的长度
        h：图像的高度（以像素为单位）
        w：图像的宽度（以像素为单位）
        '''
        bs, n, c, h, w = images.size()
        V = torch.zeros((bs, self.N + 1, 49, self.d_model)) #49 表示每张图像经过池化后的特征包含的空间位置数
        if torch.cuda.is_available():
            V = V.cuda()

        for i in range(self.N + 1):
            # Last conv layer feature map
            # previous image feature: bs x feature_size x 7 x 7
            A = self.feature(images[:, i]) #images[:, i]表示取第i列的数据
            # bs x 49 x feature_size
            feat = A.view(bs, A.size(1), -1).transpose(1, 2) #通过全连接层 self.affine_as[i]，将其映射到大小为 (bs, 49, self.d_model) 的隐藏层表示
            # bs x 49 x d_model
            feat = F.relu(self.affine_as[i](self.dropout(feat)))
            # bs x (N+1) x 49 x d_model
            V[:, i] = feat  #V[:, i] 表示第 i 个位置的特征表示。这样，V 就包含了整个图像序列的特征表示
        return V

# Text Encoder
class TNN(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.1, num_layers=1,
                 vocab_size=1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.posit_embedding = nn.Embedding(1000, embed_dim)
        self.transform = nn.ModuleList(
            [TransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    #todo 确认pad_id的值
    def forward(self, txt=None, pad_mask=None, att_mask=None):
        if txt != None:
            if pad_mask == None:
                pad_mask = (txt == 0)  # (B,L) pad_id=3的词汇，则在生成pad_mask时，对应的位置会被标记为无效（0）
            posit_index = torch.arange(txt.shape[1]).unsqueeze(0).repeat(txt.shape[0], 1).to(
                txt.device)  # (B,L)
            posit_embed = self.posit_embedding(posit_index)  # (B,L,E)
            token_embed = self.token_embedding(txt)  # (B,L,E)
            final_embed = self.dropout(token_embed + posit_embed)  # (B,L,E)
        else:
            raise ValueError('txt or token_embed must not be None')

        for i in range(len(self.transform)):
            final_embed = self.transform[i](final_embed, pad_mask, att_mask)[0]

        return final_embed  # (B,L,E)

class EncoderTXT(nn.Module):
    """encode conditional report"""

    def __init__(self, vocab_size, embed_size, hidden_size, N):
        super(EncoderTXT, self).__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.tnn = TNN(embed_dim=embed_size, num_heads=NUM_HEADS, fwd_dim=hidden_size, dropout=DROPOUT, num_layers=NUM_LAYERS,
                  vocab_size=vocab_size)

    def forward(self, input):
        # encode
        # embedding: bs x N x tlen -> bs x N x tlen x d_model
        embed_report = self.embed(input) #将输入的文本序列转换为嵌入向量，嵌入向量可以将文本序列中的每个单词表示为一个固定长度的向量，有助于神经网络处理文本数据
        #嵌入层的作用是将输入的离散化的单词转换为连续的嵌入向量，这些嵌入向量可以捕捉单词之间的语义关系

        bs, N, tlen, d_model = embed_report.size() #(batch_size, 序列个数, 序列长度, embedding向量长度)
        align_report = torch.zeros((bs, self.N, tlen, d_model)) #创建了一个大小为(bs, N, tlen, d_model)的四维张量，所有元素都被初始化为0
        if torch.cuda.is_available():
            align_report = align_report.cuda()

        for i in range(self.N):
            #  bs x tlen x d_model
            align_report_ = self.tnn(txt=input[:, i]) # (B,L,E)
            align_report[:, i] = align_report_

        return align_report


class ConditionText(nn.Module):
    """generate conT from conditional
    """

    def __init__(self, d_model, n_head=8, d_ff=2048, dropout=0.1, N=1):
        super(ConditionText, self).__init__()
        self.N = N
        self.attI = MultiHeadAttention(n_head, d_model, dropout)
        self.attT = MultiHeadAttention(n_head, d_model, dropout)

    def forward(self, hiddents, V, T):
        Vs, Ts = [], []
        curr_vf = V[:, -1]
        for i in range(self.N):
            prev_vf = V[:, i]
            prev_tf = T[:, i]

            delta_vf = curr_vf - prev_vf  # bs x 49 x d_model
            #  bs x 49 x d_model
            Vs.append(delta_vf)
            Ts.append(prev_tf)
        # bs x M x 49 x d_model -> bs x M*49 x d_model
        DeltaV = torch.cat(Vs, dim=1)
        T_ = torch.cat(Ts, dim=1)
        conI = self.attI(hiddents, DeltaV, DeltaV)
        conT = self.attT(hiddents, T_, T_)

        return conI, conT


# Caption Decoder
class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, N=1, v_size=49,
                 num_layers=6, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(Decoder, self).__init__()

        self.N = N
        # word embedding
        self.caption_embed = nn.Embedding(vocab_size, embed_size)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers
        )
        self.condition = ConditionText(embed_size, n_head=nhead, d_ff=dim_feedforward, N=N)
        # Attention Block
        self.attention = MultiHeadAttention(heads=nhead, d_model=hidden_size, dropout=dropout)
        self.mlp = nn.Linear(hidden_size * 4, vocab_size)
        self.dropout = nn.Dropout(dropout)

    # images, prev_repo, target
    def forward(self, V, T, captions):
        # Word Embedding, bs x len x d_model
        embeddings = self.caption_embed(captions)
        # bs x 49 x d_model
        _, curr_vf = V[:, 0], V[:, -1]
        # bs x 1 x d_model
        v_a = torch.mean(curr_vf, dim=1, keepdim=True)
        # bs x len x d_model*2
        x = torch.cat((embeddings, v_a.expand_as(embeddings)), dim=2)

        # Transformer Decoder Block
        # tgt = x.permute(1, 0, 2)
        tgt = embeddings.permute(1, 0, 2)
        memory = curr_vf.permute(1, 0, 2)
        #Transformer解码器模块中，如果没有提供掩码，则默认使用全1矩阵作为掩码，表示模型可以使用所有的目标序列信息进行预测
        #用于屏蔽目标序列中当前时间步之后的位置，以避免模型在预测时使用未来的信息
        #产生一个上三角矩阵，下三角的值全为1，上三角的值权威0，对角线也是1。
        tgt_mask = nn.Transformer().generate_square_subsequent_mask(x.size(1)).to(x.device)
        tgt_mask = tgt_mask.unsqueeze(0).expand(512, -1, -1) #tgt_mask的形状应该是(seq_len, seq_len)，其中seq_len是目标序列的长度
        memory_mask = None
        #tgt：解码器模块的目标序列，目标序列通常是指图像描述中的文字序列 memory：编码器模块的输出序列，图像特征
        #目标序列通常是指图像描述中的文本序列。在图像描述任务中，我们希望生成一个与输入图片相对应的文字序列，因此我们需要将输入图片的视觉信息融入到生成的文本序列中，以保证生成的描述与输入图片相符。
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        # Condition Text
        conI, conT = self.condition(output.permute(1, 0, 2), V, T)
        # Attention Block
        ctx = self.attention(output.permute(1, 0, 2), curr_vf, curr_vf)
        # Concatenation and Final Score
        output = torch.cat([output.permute(1, 0, 2), ctx, conI, conT], dim=2)
        scores = self.mlp(self.dropout(output))
        return scores


# Whole Architecture with Image Encoder and Caption decoder
class Encoder2Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, N):
        super(Encoder2Decoder, self).__init__()

        # Image CNN encoder
        self.encoder_image = EncoderCNN(embed_size, hidden_size, N)

        # Concept encoder
        self.encoder_concept = EncoderTXT(vocab_size, embed_size, hidden_size, N)

        # Caption Decoder
        self.decoder = Decoder(embed_size, vocab_size, hidden_size, N)

        # Share the weight matrix between caption & concept word embeddings
        self.encoder_concept.embed.weight = self.decoder.caption_embed.weight

        assert embed_size == hidden_size, "The values of embed_size and hidden_size should be equal."

    # images, target, prev_repo
    def forward(self, images, captions, image_concepts, lengths, basic_model):
        # imag -> V : bs x C x H x W -> bs x 2 x 49 x d_model
        # concept -> T: bs x tlen -> bs x tlen x d_model
        V = self.encoder_image(images)
        T = self.encoder_concept(image_concepts)

        # Language Modeling on word prediction
        # bs x len x vocab_size
        scores = self.decoder(V, T, captions)

        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence(scores, lengths, batch_first=True)

        return packed_scores
