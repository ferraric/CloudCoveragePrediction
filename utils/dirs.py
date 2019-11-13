import os
from os import listdir
from os.path import isfile, join

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def list_files_in_directory(directory):
    """Takes the path to a directory and lists all files in it
    :param: directory
    :return list of file paths
    """
    return [f for f in listdir(directory) if isfile(join(directory, f))]
