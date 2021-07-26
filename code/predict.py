import os
import torch
from PIL import Image
from torchvision import transforms
from network.vit_model import vit_base_patch16_224_in21k as create_model
from utils import preprocess


test_dir = './raw_dataset/test_pic/'

def main():
    preprocess()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    model = create_model(num_classes=10, has_logits=False).to(device)
    model_weight_path = "./models/model-11.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        total = 0
        drink_total = 0
        for q in os.listdir(test_dir):

            predict_dic = {
                'safe': 0,
                'phone': 0,
                'drink': 0,
                'back': 0,
                'sing_hand': 0,
            }

            pic_name_list = [1] * 70

            for j in os.listdir(os.path.join(test_dir, q)):
                pic_name_list[int(j.split('.')[0])] = j

            for i in pic_name_list:
                img = Image.open(test_dir + q + '/' + i)
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0)

                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

                if predict_cla == 0 or predict_cla == 9:
                    predict_dic['safe'] += 1
                elif predict_cla == 6:
                    predict_dic['drink'] += 1
                elif predict_cla == 7:
                    predict_dic['back'] += 1
                elif predict_cla == 8 or predict_cla == 5:
                    predict_dic['sing_hand'] += 1
                elif predict_cla == 1 or predict_cla == 2 or predict_cla == 3 or predict_cla == 4:
                    predict_dic['phone'] += 1

            predict_sort = sorted(predict_dic.items(), key=lambda item: item[1])
            if predict_sort[-1][0] == 'safe' and predict_sort[-1][1] >= 50:
                lable = 'safe'
            elif predict_sort[-1][0] == 'sing_hand':
                if predict_sort[-2][1] != 0:
                    lable = predict_sort[-2][0]
                else:
                    lable = 'drink'
            else:
                lable = predict_sort[-1][0]

            total += 1
            if lable == 'drink':
                drink_total += 1
            print(lable)



if __name__ == '__main__':
    main()
