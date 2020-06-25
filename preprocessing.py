import yaml
import os
import cv2


def preprocess_replayattack():
    # create necessary dir for output
    outdir = './dataset/replayattack_prep'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    real_outdir = os.path.join(outdir, 'real')
    if not os.path.exists(real_outdir):
        os.makedirs(real_outdir)
    attack_outdir = os.path.join(outdir, 'attack')
    if not os.path.exists(attack_outdir):
        os.makedirs(attack_outdir)

    # generate fake frames
    datasetdir = './dataset/replayattack/train'
    fake_videos_dir = os.path.join(datasetdir, 'attack', 'hand') # we only take the attack videos in folder hand
    for video_name in os.listdir(fake_videos_dir):
        client = video_name.split('_')[2]
        outdir_4_client = os.path.join(attack_outdir, client)
        if not os.path.exists(outdir_4_client):
            os.makedirs(outdir_4_client)
        video_path = os.path.join(fake_videos_dir, video_name)
        print('opening ', video_path)
        cap = cv2.VideoCapture(video_path)
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        # Read until video is completed
        frame_idx = 0
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                outpath_4_client = os.path.join(outdir_4_client, video_name.split('.')[0]+'_{}.jpg'.format(frame_idx))
                if frame_idx % 100 == 0: # save frame to file in the period of 100 frames
                    cv2.imwrite(outpath_4_client, frame)
                frame_idx += 1
            else:
                break
        # When everything done, release the video capture object
        cap.release()

    # generate real frames
    real_videos_dir = os.path.join(datasetdir, 'real')
    for video_name in os.listdir(real_videos_dir):
        client = video_name.split('_')[0]
        outdir_4_client = os.path.join(real_outdir, client)
        if not os.path.exists(outdir_4_client):
            os.makedirs(outdir_4_client)
        video_path = os.path.join(real_videos_dir, video_name)
        print('opening ', video_path)
        cap = cv2.VideoCapture(video_path)
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        # Read until video is completed
        frame_idx = 0
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                outpath_4_client = os.path.join(outdir_4_client,
                                                video_name.split('.')[0] + '_{}.jpg'.format(frame_idx))
                if frame_idx % 50 == 0:  # save frame to file in the period of 100 frames
                    cv2.imwrite(outpath_4_client, frame)
                frame_idx += 1
            else:
                break
        # When everything done, release the video capture object
        cap.release()


if __name__ == '__main__':
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    if cfg['dataset']['name'] == 'replayattack':
        preprocess_replayattack()
