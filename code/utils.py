import sys
import torch
from tqdm import tqdm
from dataLoaders.video_dataset import *
import re
import mypath
import config

args = config.parse_opt()


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.long()
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        

        
        loss = loss_function(pred, labels.to(device))
        

        
        loss = loss/16
        # 2.2 back propagation
        loss.backward()


        if((step+1)%16)==0:
            accu_loss += (loss.detach() * 16)
            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)
            
                # 3. update parameters of net
        # optimizer the net
            optimizer.step()        # update parameters of net
            optimizer.zero_grad() 


    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.long()
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        
        loss = loss/16
        
        accu_loss += loss
        
        if((step+1)%16)==0:         
        
            accu_loss += loss * 16
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


root_dir, output_dir = Path.db_dir('test')
def process_video(video):
    video_name = video.split('/')[-1]
    video_name = video_name.split('.')[0]

    root_dir, output_dir = Path.db_dir('test')
    if not os.path.exists(os.path.join(output_dir, video_name)):
        os.mkdir(os.path.join(output_dir, video_name))


    capture = cv2.VideoCapture(video)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Make sure splited video has at least 16 frames
    EXTRACT_FREQUENCY = 1
    if frame_count // EXTRACT_FREQUENCY < 70:
        print('Error. The clip only has %d frames which is less than 70 frames.' % frame_count)
        return None

    count = 0
    i = 0
    retaining = True

    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            if (frame_height != args.resize_height) or (frame_width != args.resize_width):
                frame = cv2.resize(frame, (args.resize_width, args.resize_height))
            cv2.imwrite(filename=os.path.join(output_dir, video_name, '{}.jpg'.format(str(i))), img=frame)
            i += 1
        count += 1
    capture.release()

def load_frames(file_dir):
    frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
    frame_count = len(frames)
    buffer = np.empty((frame_count, args.resize_height, args.resize_width, 3), np.dtype('float32'))
    for i, frame_name in enumerate(frames):
        frame = np.array(cv2.imread(frame_name)).astype(np.float64)
        buffer[i] = frame
    return buffer

def normalize(buffer):
    for i, frame in enumerate(buffer):
        frame -= np.array([[[0.5,0.5,0.5]]])
        buffer[i] = frame

    return buffer


def prepare_data():
    for pic_file in sorted(os.listdir(output_dir)):
        fnames = os.path.join(output_dir, pic_file)
    buffer = load_frames(fnames)
    buffer = normalize(buffer)
    return torch.from_numpy(buffer)

if __name__ == '__main__':
    process_video('./raw_dataset/test/test_61.mp4')