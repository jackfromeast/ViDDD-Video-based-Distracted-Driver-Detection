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
            frame_feature = self.vit(x[i])
            frame_list.append(frame_feature.unsqueeze(0))
        gru_input = torch.cat(frame_list,dim=1)
        gru_output, _ = self.gru(gru_input)
        out = self.dense(gru_output)
        return out























































        # if args.weights != "":
        #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        #     weights_dict = torch.load(args.weights, map_location=device)
        #     # 删除不需要的权重
        #     del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        #         else ['head.weight', 'head.bias']
        #     for k in del_keys:
        #         del weights_dict[k]
        #     print(model.load_state_dict(weights_dict, strict=False))
        #
        # if args.freeze_layers:
        #     for name, para in model.named_parameters():
        #         # 除head, pre_logits外，其他权重全部冻结
        #         if "head" not in name and "pre_logits" not in name:
        #             para.requires_grad_(False)
        #         else:
        #             print("training {}".format(name))