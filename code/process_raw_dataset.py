import os
import re
import json
import cv2
import shutil


def rename_for_DMD(filespath):
    """
    Remove the timestamp in the raw video name
    because it may cause fileNotExists Error in ffmpeg reading
    """
    files = os.listdir(filespath)
    
    for filename in files:
        items = re.split("[_.]", filename)

        os.rename(filespath+'/'+filename, filespath+'/'+'_'.join(items[:3])+'.' + items[-1])


def cut_video(v_path, save_path, frames_n=70):
    """
    Cut the video into (default) 70 frames videoclips and save it

    v_path: "./raw_dataset/DMD-lite/gA_1_s1.mp4"
    save_path: "./processed_data/DMD_clips_70/"
    frames_n: 70
    """

    #获取视频对象
    cap = cv2.VideoCapture(v_path)

    #判断视频是否能顺利打开读取，True为能打开，False为不能打开
    if not cap.isOpened():
        print("The video can't open successfully.")
        return None
    
    # 获取视频信息
    rate = cap.get(5)  #获取视频的帧速率
    frameNumber = int(cap.get(7)) #获取视频文件的总帧数
    duration = frameNumber/rate #视频总帧数除以帧速率等于视频时间，除以60之后的单位就是分钟
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) #获得视频帧宽和帧高
    fps = int(rate)

    actions_dict = setup_annotation_file(v_path)

    frame_count = 0 # 正在处理第 frame_count 个 frame
    clip_count = 0 # 正在保存第 clip_count 个 clip
    while(True):
        # 检查是否能正常读取帧 以及 当前帧的图像
        success, a_frame = cap.read()

        if success:
            frame_count += 1  # 注意这里从1开始计数 没有第0帧

            if (frame_count % frames_n == 1):
                clip_count += 1

                video_name = v_path.split('/')[-1] # video_name is expected in "gA_1_s1.mp4"
                folder_name = video_name.split('.')[0]
                folder_path = save_path+folder_name

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                # 在ann文件中读取该clip包含的动作
                action = check_action(actions_dict, frame_count)
                clip_path = folder_path + '/' + folder_name + '_' + str(clip_count) + '_' + action + '.mp4'

                videoWriter = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height))) # 保存文件名、编码器、帧率、视频宽高

                # 写入帧图像
                videoWriter.write(a_frame)

            else:
                videoWriter.write(a_frame)

        else:  
            print('Have processed %d frames within video %s, generated %d clips.' % (frame_count, video_name, clip_count))
            break


def setup_annotation_file(v_path):
    """
        setup the annnotion dict only once
    """
    json_file_path = v_path.split('.')[0] + '.json'

    with open(json_file_path, 'r') as f:
        ann_dict = json.load(f)
    
    actions_dict = {}
    for ann in ann_dict['vcd']['actions'].values():
        if ann['type'].split('/')[0] == 'driver_actions':
            action_name = ann['type'].split('/')[-1]

            intervals = []
            for interval in ann['frame_intervals']:
                start = interval['frame_start']
                end = interval['frame_end']

                intervals.append((start, end))
            
            actions_dict[action_name] = intervals
        else:
            continue
    
    return actions_dict


def check_action(actions_dict, frame_s):
    """
    check the actions in the frames between (frame_s, frame_s+70) through actions dict

    actions_dict: {'action_name':[(interval_s, interval_e), (), ...]}
    frame_s: the start frame of the clip
    """
    temp = [] # Incase the clips include more than one action
    for action_name, intervals in actions_dict.items():
        for interval in intervals:
            if frame_s >= interval[0] and frame_s+70 <= interval[1]:
                return action_name
            
            elif frame_s+70 <= interval[0] or frame_s >= interval[1]:
                continue

            # 如果一个clip中包含多个动作，则添加到temp中
            else:
                temp.append(action_name)

    if temp == ['safe_drive', 'unclassified']:
        return 'safe_drive'

    subtraction = list(set(temp).difference(set(['safe_drive', 'unclassified'])))

    if len(subtraction) == 1:
        return subtraction[0]
    else:
        return 'unclassified'



def process_raw_dataset(raw_dataset_path, save_path):
    """
        Main function used to process raw DMD dataset.

        raw_dataset_path: "./raw_dataset/DMD-lite/"
        save_path: "./processed_data/DMD_clips_70/"
    """
    for video_name in os.listdir(raw_dataset_path):
        if video_name.endswith('.mp4'):
            cut_video(raw_dataset_path + video_name, save_path)
            print("Video %s is done." % video_name)
    


def restructure(processed_dataset_path):
    """
    Restructure the dataset like the form as follows.
        UCF-101
        ├── ApplyEyeMakeup
        │   ├── v_ApplyEyeMakeup_g01_c01.avi
        │   └── ...
        ├── ApplyLipstick
        │   ├── v_ApplyLipstick_g01_c01.avi
        │   └── ...
        └── Archery
        │   ├── v_Archery_g01_c01.avi
        │   └── ...
    """
    for root_path, dirs, files in os.walk(processed_dataset_path):

        for file_name in files:
            file_path = root_path + '/' + file_name

            class_name = re.split('[_.]', file_name)[4:-1]
            class_name = '_'.join(class_name)
            class_path = processed_dataset_path + class_name

            if not os.path.exists(class_path):
                os.makedirs(class_path)
            
            old_file = file_path
            new_file = class_path + '/' + file_name

            shutil.copyfile(old_file, new_file) # incase something is wrong
            


if __name__ == "__main__":
    # rename_for_DMD('./raw_dataset/DMD-lite')
    
    # cut_video('raw_dataset/DMD-lite/gA_2_s1.mp4', 'processed_dataset/DMD-clips-70/', 70)
    # process_raw_dataset('raw_dataset/DMD-lite/', 'processed_dataset/DMD-clips-70/')

    # restructure('processed_dataset/DMD-clips-70/')