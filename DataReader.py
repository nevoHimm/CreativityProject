import cv2
import numpy as np
import pandas as pd
import glob
import os
import featuresExtractor
import pathlib

IGNORE_TAG = -1

FRAME_COLS = ['pAngry', 'pDisgusted', 'pAfraid', 'pHappy', 'pSad', 'pSurprised', 'pNeutral', 'eyeDir', 'faceAppeared',
              'ConcentrationLvl']
SAMPLE_COLS = ['blinkRate', 'movementRate', 'eyeDir', 'pAngry', 'pDisgusted', 'pAfraid', 'pHappy', 'pSad', 'pSurprised',
               'pNeutral', 'faceAppeared', 'ConcentrationLvl']


def get_samples(vid, desired_fps, train=True):
    # TODO delete sample_df param
    """
    tag frames from video, in desired fps
    :return - an np array holding each frame and it's tag
    """
    cur_fps = np.round(vid.get(cv2.CAP_PROP_FPS))
    sel_frame = np.int(cur_fps / desired_fps)
    is_first_frame = True
    frame_count, tagged_frame_count = 0, 0

    samples_df = pd.DataFrame(columns=SAMPLE_COLS)

    while True:
        ret, frame = vid.read()
        if ret:
            if frame_count % sel_frame == 0:
                featuresExtractor.get_frame_features(frame, is_first_frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            frame_count += 1
        else:
            break

    # TODO add remaining frames as last sample?
    # sample_data = calc_sample_data(tagged_frames, blinks_per_sample, move_per_sample, sample_length)
    # samples_df = samples_df.append(sample_data, ignore_index=True)

    vid.release()
    cv2.destroyAllWindows()
    return samples_df


def calc_sample_data(tagged_frames, blinks_per_sample, move_per_sample, sample_length):
    # convert to blink and movement rate
    blink_rate = blinks_per_sample / sample_length
    move_rate = move_per_sample / sample_length

    # average other features from entire sample time
    eyeDir = tagged_frames['eyeDir'].value_counts().idxmax()  # get most looked at direction
    pAngry = tagged_frames['pAngry'].mean()
    pDisgusted = tagged_frames['pDisgusted'].mean()
    pAfraid = tagged_frames['pAfraid'].mean()
    pHappy = tagged_frames['pHappy'].mean()
    pSad = tagged_frames['pSad'].mean()
    pSurprised = tagged_frames['pSurprised'].mean()
    pNeutral = tagged_frames['pNeutral'].mean()
    faceAppeared = tagged_frames[
        'faceAppeared'].value_counts().idxmax()  # was there a face in the frame for most of the sample
    ConcentrationLvl = tagged_frames[
        'ConcentrationLvl'].value_counts().idxmax()  # what was the common concentration level

    sample_data = pd.DataFrame(
        data={'blinkRate': [blink_rate], 'movementRate': [move_rate], 'eyeDir': [eyeDir], 'pAngry': [pAngry],
              'pDisgusted': [pDisgusted], 'pAfraid': [pAfraid], 'pHappy': [pHappy], 'pSad': [pSad],
              'pSurprised': [pSurprised], 'pNeutral': [pNeutral], 'faceAppeared': [faceAppeared],
              'ConcentrationLvl': [ConcentrationLvl]})
    return sample_data


def get_frame_data(features, tag):
    return pd.DataFrame(
        {'pAngry': [features[0]], 'pDisgusted': [features[1]], 'pAfraid': [features[2]],
         'pHappy': [features[3]], 'pSad': [features[4]], 'pSurprised': [features[5]],
         'pNeutral': [features[6]], 'eyeDir': [features[7]], 'faceAppeared': [features[8]],
         'ConcentrationLvl': [tag]})


def get_tag(tags, frame_timestamp):
    """
    get tag according to time
    :param tags: df of tags by time in video
    :param time: time of frame in video
    :return: the tag in the relevant moment
    """
    all_tags_before = tags.loc[tags['time'] <= to_datetime(frame_timestamp)]
    tag = all_tags_before['tag'].iloc[int(max(0, all_tags_before.shape[0] - 1))]
    return tag


def to_datetime(frame_timestamp):
    dt = pd.to_datetime(frame_timestamp * 1000000 * 60)
    return dt.combine(pd.to_datetime('today').date(), dt.time())


def count_frames(video, override=False):
    # grab a pointer to the video file and initialize the total
    # number of frames read
    total = 0

    # if the override flag is passed in, revert to the manual
    # method of counting frames
    if override:
        total = count_frames_manual(video)

    # otherwise, let's try the fast way first
    else:
        # lets try to determine the number of frames in a video
        # via video properties; this method can be very buggy
        # and might throw an error based on your OpenCV version
        # or may fail entirely based on your which video codecs
        # you have installed
        try:
            total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        # uh-oh, we got an error -- revert to counting manually
        except:
            total = count_frames_manual(video)

    # release the video file pointer
    video.release()

    # return the total number of frames in the video
    return total


def count_frames_manual(video):
    # initialize the total number of frames read
    total = 0

    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break

        # increment the total number of frames read
        total += 1

    # return the total number of frames in the video file
    return total


def read_data(path, fps, sample_len, train=True):
    data = pd.DataFrame(columns=SAMPLE_COLS)

    if not train:
        all_videos = glob.glob(os.path.join(path, "*.mp4"))  # assuming mp4 format
        for vid_path in all_videos:
            df = process_video(path, vid_path, os.path.splitext(os.path.basename(vid_path))[0], 1, fps, sample_len,
                               train=False)
            data = data.append(df, ignore_index=True)
        return data.drop(['ConcentrationLvl'], axis=1)

    # read all csv tag files
    all_csv_files = glob.glob(os.path.join(path, "*.csv"))
    for filename in all_csv_files:
        # if os.path.splitext(os.path.basename(filename))[0] == "Alon":  # TODO delete
        # read file, extract features and save as one df
        print(filename)  # TODO delete
        if not os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(filename))[0] + ".pkl")):
            tags = pd.read_csv(filename, index_col=None, header=0, parse_dates=['time'])
            vid_path = os.path.join(path, os.path.splitext(os.path.basename(filename))[0] + ".mp4")
            df = process_video(path, vid_path, filename, tags, fps, sample_len)
            # df.to_pickle(os.path.join(path, os.path.splitext(os.path.basename(filename))[0] + ".pkl"))
        else:
            df = pd.read_pickle(os.path.join(path, os.path.splitext(os.path.basename(filename))[0] + ".pkl"))
        data = data.append(df, ignore_index=True)
        # break
    return data


def process_video(path, vid_path, fps, train=True):
    vid = cv2.VideoCapture(vid_path)
    df = get_samples(vid, fps, train=train)
    vid.release()
    return df


print(featuresExtractor.get_emotion_count())
process_video(pathlib.Path(__file__).parent.absolute(),
              r"C:\Users\user\Desktop\לימודים\שנה 2\סמסטר ב\הקורס של מיכאל\יצירתיות\פייתון\עמית.mp4",
              10)
print(featuresExtractor.get_emotion_count())


# תמי
# עמית
