"""
Loads the GTSRB database from the hard drive.
"""

import os

import pandas as pd


def load_traffic_sign_database(path_to_directory: str) -> pd.DataFrame:
    """

    :param path_to_directory: Points to the root directory of the traffic sign folder structure
    :return: DataFrame containing the paths to all images
    """
    traffic_sign_class_dfs = []

    if not os.path.exists(path_to_directory):
        raise Exception("The following path does not exist: '{p}'".format(p=path_to_directory))

    if "GTSRB" in os.listdir(path_to_directory):
        path_to_directory = os.path.join(path_to_directory, "GTSRB")
    if "Final_Training" in os.listdir(path_to_directory):
        path_to_directory = os.path.join(path_to_directory, "Final_Training")
    if "Images" in os.listdir(path_to_directory):
        path_to_directory = os.path.join(path_to_directory, "Images")

    for traffic_sign_class in log_progress(os.listdir(path_to_directory)):
        if os.path.isfile(os.path.join(path_to_directory, traffic_sign_class)):
            continue  # e.g. the readme file
        meta_info_file_name = "GT-{class_id}.csv".format(class_id=traffic_sign_class)
        path_to_meta_info_file = os.path.join(path_to_directory, traffic_sign_class, meta_info_file_name)
        if not os.path.exists(path_to_meta_info_file):
            raise Exception("The following file does not exist: '{p}'".format(p=path_to_meta_info_file))
        df = pd.read_csv(path_to_meta_info_file, delimiter=";")
        df.columns = [col.replace(".", "_") for col in df.columns]
        images = []
        images = df.Filename.apply(lambda file_name : os.path.join(os.path.dirname(path_to_meta_info_file), file_name))
        df = df.assign(path_to_image=images)
        traffic_sign_class_dfs.append(df)

    return pd.concat(traffic_sign_class_dfs)


def log_progress(sequence, every=None, size=None, name='Items'):
    """
    taken from https://github.com/alexanderkuk/log-progress

    """
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)  # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )
