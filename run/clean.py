import os
from pandas import read_csv

from run.utilities import yn_input

from problem import model_dir, results_dir

def delete_files(files_to_clean):
    w = os.get_terminal_size().columns
    print('\n' + '-'*w + '\n')
    for file_path in files_to_clean:
        print(file_path)
    print('\n' + '-'*w + '\n')
    if yn_input('Delete the files listed above?'):
        for file_path in files_to_clean:
            os.remove(file_path)

info_path = os.path.join(model_dir, 'model_info.csv')

try:
    timestamps = read_csv(info_path, usecols=['timestamp']).dropna()
    timestamps = timestamps.to_numpy().flatten().astype(str)
except:
    timestamps = []

files_to_clean = []

for file_name in os.listdir(model_dir):
    # Models are saved as timestamp.pkl
    if file_name[-4:] == ".pkl":
        if file_name[:-4] not in timestamps:
            files_to_clean.append(os.path.join(model_dir, file_name))
        else:
            pass

n_clean = len(files_to_clean)
if n_clean:
    print('\nThe following %d saved models will be deleted:' % n_clean)
    delete_files(files_to_clean)

results_dir = os.path.join(results_dir, 'monte_carlo')

files_to_clean = []

# Tests results are saved as test_timestamp.mat
for file_name in os.listdir(results_dir):
    if file_name[-4:] == ".mat" and file_name[5:-4] not in timestamps:
        files_to_clean.append(os.path.join(results_dir, file_name))

n_clean = len(files_to_clean)
if n_clean:
    print('\nThe following %d Monte Carlo tests will be deleted:' % n_clean)
    delete_files(files_to_clean)
