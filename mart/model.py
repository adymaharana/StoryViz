import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CAPTION_CNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(CAPTION_CNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(5, 32, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 3, 3), stride=3),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d((3, 3), stride=3),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d((3, 3), stride=3),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d((3, 3), stride=3),
        )

        self.fc1 = nn.Linear(128, 128)
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)


    def forward(self, video_images):
        """Extract feature vectors from input images."""
        #print ('image feature size before unsample:', images.size())
        m = nn.Upsample(size=(224, 224), mode='bilinear')
        images = video_images.view(-1, video_images.shape[-3], video_images.shape[-2], video_images.shape[-1])
        upsampled_images = m(images)

        #print ('image feature size after unsample:', unsampled_images.size())
        features = self.resnet(upsampled_images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))

        features = features.view(images.shape[0], -1)
        out = self.conv_layer1(video_images).squeeze(-3)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)

        out = out.unsqueeze(1).repeat(1, 5, 1).view(-1, out.shape[-1])
        return torch.cat([features, out], dim=-1)

class CAPTION_RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(CAPTION_RNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    # def forward(self, features, captions, cap_lens):
    #     """Decode image feature vectors and generates captions."""
    #     # print ('feature.size():', features.size()) #(6L, 256L)
    #     # print ('captions.size():', captions.size()) # (6L, 12L)
    #     # print ('embeddings.size:',embeddings.size()) #(6L, 12L, 256L)
    #     embeddings = self.embed(captions)
    #     embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
    #     packed = pack_padded_sequence(embeddings, cap_lens.data.tolist(), batch_first=True)
    #     outputs, hidden = self.lstm(packed)
    #     output = self.linear(outputs[0])   # (batch size, vocab_size)
    #     return output, hidden, outputs     # words embedding, sentence embedding

    def forward(self, features, captions, cap_lens):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, cap_lens, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        unpacked, seq_lengths = pad_packed_sequence(hiddens, batch_first=True)
        outputs = self.linear(unpacked)
        return outputs, seq_lengths

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


#
#
# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         super(EncoderCNN, self).__init__()
#         resnet = models.resnet50(pretrained=True)
#         for param in resnet.parameters():
#             param.requires_grad_(True)
#         modules = list(resnet.children())[:-1]
#         self.resnet = nn.Sequential(*modules)
#         self.embed = nn.Linear(resnet.fc.in_features, embed_size)
#
#     def forward(self, images):
#         features = self.resnet(images)
#         features = features.view(features.size(0), -1)
#         features = self.embed(features)
#         return features
#
#
# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super().__init__()
#         self.embedding_layer = nn.Embedding(vocab_size, embed_size)
#
#         self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
#                             num_layers=num_layers, batch_first=True, bidirectional=True)
#
#         self.linear = nn.Linear(hidden_size, vocab_size)
#
#     def forward(self, features, captions):
#         captions = captions[:, :-1]
#         embed = self.embedding_layer(captions)
#         embed = torch.cat((features.unsqueeze(1), embed), dim=1)
#         lstm_outputs, _ = self.lstm(embed)
#         out = self.linear(lstm_outputs)
#
#         return out
#
#     def sample(self, inputs, states=None, max_len=20):
#         " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
#         output_sentence = []
#         for i in range(max_len):
#             lstm_outputs, states = self.lstm(inputs, states)
#             lstm_outputs = lstm_outputs.squeeze(1)
#             out = self.linear(lstm_outputs)
#             last_pick = out.max(1)[1]
#             output_sentence.append(last_pick.item())
#             inputs = self.embedding_layer(last_pick).unsqueeze(1)
#
#         return output_sentence
#
# class ImageConvEncoder(nn.Module):
#     def __init__(self, embed_size):
#         super().__init__()
#         ndf = 16
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             # state size (ndf * 8) x 4 x 4)
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 16),
#             # state size (ndf * 8) x 4 x 4)
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.embed = nn.Linear(ndf*16*49, embed_size)
#
#     def forward(self, images):
#         features = self.encoder(images)
#         features = features.view(features.size(0), -1)
#         features = self.embed(features)
#         return features
#
#
# class ParagraphDecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, seq_len=5, num_layers=1):
#         super().__init__()
#         self.embedding_layer = nn.Embedding(vocab_size, embed_size)
#         self.img_lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
#                                   num_layers=num_layers, batch_first=True)
#         self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
#                             num_layers=num_layers, batch_first=True)
#
#         self.linear = nn.Linear(hidden_size, vocab_size)
#         self.seq_len = seq_len
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#
#     def forward(self, img_features, captions):
#
#         img_features = img_features.view(-1, self.seq_len, self.embed_size)
#         img_lstm_outputs, _ = self.img_lstm(img_features)
#         img_lstm_outputs = img_lstm_outputs.contiguous().view(-1, self.hidden_size)
#
#         captions = captions[:, :-1]
#         captions_embeds = self.embedding_layer(captions)
#
#         lstm_inputs = torch.cat((img_lstm_outputs.unsqueeze(1), captions_embeds), dim=1)
#         lstm_outputs, _ = self.lstm(lstm_inputs)
#         out = self.linear(lstm_outputs)
#         return out
#
#     def sample(self, inputs, states=None, max_len=20):
#
#         inputs = inputs.view(-1, self.seq_len, self.embed_size)
#         inputs, _ = self.img_lstm(inputs)
#         inputs = inputs.contiguous().view(-1, self.hidden_size).unsqueeze(1)
#
#         output_sentence = []
#         for i in range(max_len):
#             lstm_outputs, states = self.lstm(inputs, states)
#             lstm_outputs = lstm_outputs.squeeze(1)
#             out = self.linear(lstm_outputs)
#             last_pick = out.max(-1)
#             last_pick = last_pick[1]
#             output_sentence.append(last_pick.cpu().numpy())
#             inputs = self.embedding_layer(last_pick).unsqueeze(1)
#
#         return output_sentence


