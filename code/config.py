import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-classes', type=int, default=14)
    parser.add_argument('--val-size', type=int, default=0.2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-type', type=str, default='video')
    # parser.add_argument('--data-path', type=str,
    #                     default="/data/flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')

    parser.add_argument('--weights', type=str, default='code/models/vit_base_patch16_224_in21k.pth',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # 视频相关
    parser.add_argument('--frames-len', type=int, default=70)


    # vit
    parser.add_argument('--vit-num-classes', type=int, default=128)
    parser.add_argument('--vit-has-logits', type=bool, default=False)

    # vvit
    parser.add_argument('--vvit-hidden-dim', type=int, default=128)
    parser.add_argument('--vvit-gru-hidden-dim', type=int, default=64)
    # parser.add_argument('--vvit-hidden-dim', type=int, default=70)


    args = parser.parse_args()
    return args