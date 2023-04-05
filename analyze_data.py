import argparse
import os
import glob

from learning_experiment import LearningExp

parser = argparse.ArgumentParser()

parser.add_argument('path', help="Path to directory with learning experiment log files")

args = parser.parse_args()



files = glob.glob(os.path.join(args.path, '*_log.txt'))


for file in files:
    print(f"Analyzing: {file}")

    lexp = LearningExp.load(file)

    data = lexp.get_all_training_data()
    print(f"Data points in all training data:", len(data))

    data_dedup = data.drop_duplicates(subset=['Shape', 'Angle'])
    N_all = len(data_dedup)
    print(f"Deduplicated data points in all training data:", N_all)

    exposure_data = lexp.get_exposure_data(3)
    print(f"Data points in all exposure 3 data:", len(exposure_data))
    exposure_data_dedup = exposure_data.drop_duplicates(subset=['Shape', 'Angle'])
    N_exposure = len(exposure_data_dedup)
    print(f"Deduplicated data points in exposure 3 data:", N_exposure)


