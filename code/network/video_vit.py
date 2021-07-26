import os
import sys
sys.path.append('./code')

import torch
from torch import nn
from network.vit_model import vit_base_patch16_224_in21k
import config

args = config.parse_opt()

class vvit(nn.Module):
    def __init__(self, vit):
        super(vvit, self).__init__()
        self.frame = args.frames_len
        self.vit = vit
        self.gru = nn.GRU(128, args.vvit_gru_hidden_dim, bidirectional=False, batch_first=True)
        self.dense = nn.Linear(args.vvit_gru_hidden_dim,args.num_classes)

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


def create_vvit(device):

    vit = vit_base_patch16_224_in21k(
            num_classes=args.vit_num_classes,
            has_logits=args.vit_has_logits
            ).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)

        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if vit.has_logits \
            else ['head.weight', 'head.bias']
            # else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(vit.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in vit.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    
    model = vvit(vit)

    return model
