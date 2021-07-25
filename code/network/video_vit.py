import torch
from torch import nn

class vvit(nn.Module):
    def __init__(self, vit, gru, frame, hidden_dim, class_num):
        super(vvit, self).__init__()
        self.frame = frame
        self.vit = vit
        self.gru = gru
        self.dense = nn.Linear(hidden_dim,class_num)

    def forward(self, x):
        frame_list = []
        x = x.transpose(1,0).contiguous()
        for i in range(self.frame):
            input = x[i].permute(0,3,1,2).contiguous()

            frame_feature = self.vit(input)

            frame_list.append(frame_feature.unsqueeze(1))

        gru_input = torch.cat(frame_list,dim=1)

        # enc_hidden[-1, :, :]
        _,gru_output = self.gru(gru_input)

        out = self.dense(gru_output[-1,:,:])
        return out




