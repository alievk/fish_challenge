import sys
import os
pjoin = os.path.join
# We assume OpenCV with FFMPEG support is build and placed in ./opencv
sys.path.insert(0, './opencv/release/lib')
import cv2
import pandas as pd


class DummyPredictor(object):
    def predict(self, frame):
        pass


class Pipeline(object):
    def __init__(self, gt_csv, video_dir, predictor=DummyPredictor):
        self.gt_csv = gt_csv
        self.video_dir = video_dir
        self.predictor = predictor

        anno = pd.read_csv(gt_csv)
        video_names_anno = set(anno.video_id)
        video_names_mp4 = set([fname.split('.')[0] for fname in os.listdir(video_dir) if fname.endswith('.mp4')])
        assert video_names_anno == video_names_mp4, 'Annotated videos and mp4 files differ'
        video_names = list(video_names_anno)
        self.anno = anno
        self.video_names = video_names

    def predict(self, video_name):
        assert video_name in self.video_names, '{} does not exist'.format(video_name)
        path = pjoin(self.video_dir, video_name + '.mp4')
        assert os.path.exists(path), '{} does not exist'.format(path)
        cap = cv2.VideoCapture(path)
        assert cap.isOpened(), 'Could not open {}'.format(path)
        print 'Processing {}'.format(path)
        iframe = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print 'Processed {} frames'.format(iframe)
                break
            iframe += 1


if __name__ == '__main__':
    # we will assume that the training data, namely training videos and ground truth csv file are under ./data directory
    data_root = './data'
    video_dir = pjoin(data_root, 'train_videos')
    gt_csv = pjoin(data_root, 'training.csv')
    assert os.path.exists(video_dir), '{} does not exists'.format(video_dir)
    assert os.path.exists(gt_csv), '{} does not exists'.format(gt_csv)

    pipeline = Pipeline(gt_csv, video_dir)
    pipeline.predict('wOOxOeyoVSnGVZSc')
