import os


class CascadeTrainer:
    def __init__(self, save_dir, label):
        positive_path = os.path.join(save_dir, 'Positive', label)
        negative_path = os.path.join(save_dir, 'Negative')

        self.pos_nums = len(os.listdir(positive_path))
        self.neg_nums = len(os.listdir(negative_path))

    def train(self):
        os.system(f'opencv_createsamples -info Dataset/Positive.txt -num {self.pos_nums} -w 64 -h 64 -vec ModelFiles/positives.vec')

        print('Training Started... \n')
        os.system(f'opencv_traincascade -data ModelFiles/ -vec ModelFiles/positives.vec -bg Dataset/Negative.txt '
                  f'-numPos {self.pos_nums} -numNeg {self.neg_nums} -numStages 10 -w 64 -h 64 -minHitRate 0.000005')
        print('Training Finished. \n')

    def test(self):
        print(os.path.exists('../ModelFiles/positives.vec'))
        os.system('opencv_createsamples -vec ModelFiles/positives.vec -w 64 -h 64 -show')