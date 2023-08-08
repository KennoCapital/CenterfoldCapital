from path import Path
import os


def get_dir_path(dir_name, file_name=None):
    path = Path(__file__).parent.parent.parent / dir_name
    if file_name:
        path = os.path.join(path, file_name)
    return path


def get_data_path(file_name=None):
    path = Path(__file__).parent.parent.parent / 'data'
    if file_name:
        path = os.path.join(path, file_name)
    return path


def get_plot_path(file_name=None):
    path = Path(__file__).parent.parent.parent / 'plots'
    if file_name:
        path = os.path.join(path, file_name)
    return path


if __name__ == '__main__':
    print(get_data_path('training_data_PUT.csv'))
