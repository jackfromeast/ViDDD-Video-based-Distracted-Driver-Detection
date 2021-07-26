import os
import torch
from PIL import Image
from torchvision import transforms
from network.vit_model import vit_base_patch16_224_in21k as create_model
from utils import process_video

from mypath import Path
import config

args = config.parse_opt()


# Load the model once
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes=10, has_logits=False).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    return model


# only predict one clip in test_dir once a time
def predict(model, video_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    process_video(video_path)
    
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    with torch.no_grad():
        # for q in os.listdir(test_dir):
        predict_dic = {
            'safe': [],
            'phone': [],
            'drink': [],
            'back': [],
            'sing_hand': [],
        }

        _, output_dir = Path.db_dir('test')
        video_folder = video_path.split('/')[-1].split('.')[0]
        
        for i in range(0, args.frames_len, 14):
            img = Image.open(output_dir +'/'+video_folder + '/' + str(i)+'.jpg')
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            predict_pos = predict[predict_cla]

            if predict_cla == 0 or predict_cla == 9:
                predict_dic['safe'].append(predict_pos)
            elif predict_cla == 6:
                predict_dic['drink'].append(predict_pos)
            elif predict_cla == 7:
                predict_dic['back'].append(predict_pos)
            elif predict_cla == 8 or predict_cla == 5:
                predict_dic['sing_hand'].append(predict_pos)
            elif predict_cla == 1 or predict_cla == 2 or predict_cla == 3 or predict_cla == 4:
                predict_dic['phone'].append(predict_pos)

        predict_sort = sorted(predict_dic.items(), key=lambda item: len(item[1]))

        if predict_sort[-1][0] == 'safe' and len(predict_sort[-1][1]) >= 3:
            label = 'safe'
        elif predict_sort[-1][0] == 'sing_hand':
            if predict_sort[-2][1] != 0:
                label = predict_sort[-2][0]
            else:
                label = 'drink'
        else:
            label = predict_sort[-1][0]
    # print(label)
    # print(max(predict_dic[label]).item())

    return(label, max(predict_dic[label]).item())
    


if __name__ == '__main__':
    model = load_model()
    print(predict(model, './raw_dataset/test/test_4.mp4'))
