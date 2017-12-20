
from learn import Learner

if __name__ == '__main__':

    reinforcement = Learner(volume_filepath = 'sample_volume.nii.gz',
        label_filepath = 'sample_volume-label.nii.gz',
        radius = 16,
        hidden_num = 16,
        learning_rate=.1,
        random_rate=.001,
        epochs=5000,
        display_step=1,
        layer_num=2,
        output_results='results.csv')

    reinforcement.run()