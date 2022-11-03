import os


def list_paths_for_files_in_dir_and_subdir(path_top_dir, ext='.dcm'):
    """
    Function to return a list of paths for all files in a top directory
    and its subdirectories.

    Args: \n
    path_top_dir
    ext: required extension for files
    """

    list_paths = []
    files=[]
    # os.walk(): For each directory in the tree rooted at directory top (including top itself),
    # it yields a 3-tuple (dirpath, dirnames, filenames).

    for dirpath, _, file_names in os.walk(path_top_dir):
        for file_name in file_names:
            fn, extension = os.path.splitext(file_name)
            if extension == ext:
                pathobj_file = os.path.join(dirpath, file_name)

                # Append to list of paths
                list_paths.append(pathobj_file)

                files.append(fn)
    return list_paths, files


    