from pathlib import Path, PurePath
import numpy as np
import cv2
import h5py
from tqdm.auto import tqdm


def get_movie_info(vfile):

    vfile = vfile.as_posix() if isinstance(vfile, PurePath) else vfile
    cap = cv2.VideoCapture(vfile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    return width, height, n_frames, fps


def read_movie_np(vfile, split=0, n_splits=1, progress=True):

    ### initialize video reader ###

    cap = cv2.VideoCapture(vfile)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    ### calculate rows to keep ###

    split_rows = int(n_rows / n_splits)
    start_row = split * split_rows

    extra_rows = n_rows % n_splits
    if extra_rows > split:
        split_rows += 1
    start_row += min(split, extra_rows)

    ### initalize np array ###

    movie_np = np.zeros(
        (
            split_rows,
            n_cols,
            n_frames,
        ),
        dtype=np.uint8,
    )

    ### loop through frames ###

    frame_it = tqdm(range(n_frames), leave=False) if progress else range(n_frames)

    for i in frame_it:
        ret, frame = cap.read()
        if ret:
            movie_np[:, :, i] = frame[:, :, 0].reshape(n_rows, n_cols)[
                start_row : (start_row + split_rows)
            ]

    return movie_np


def movie_to_h5(vfile, splits=1, progress=True, overwrite=False):

    ### check for existing file ###

    out_file = Path(vfile).with_suffix(".h5")
    if out_file.is_file():
        if not overwrite:
            print(
                f"hdf5 video file = {out_file.as_posix()} already exists. Please pass overwrite=False to overwrite this file."
            )
            return out_file

    ### get movie dimensions ###

    n_cols, n_rows, n_frames, _ = get_movie_info(vfile)

    ### create memory mapped file and array ###

    f = h5py.File(out_file, mode="w")
    movie_h5 = f.create_dataset(
        "data",
        shape=(n_rows, n_cols, n_frames),
        chunks=(1, 1, n_frames),
        dtype="uint8",
    )

    if splits > 1:

        cur_row = 0
        for i in range(splits):

            if progress:
                print(f"split = {i+1} / {splits}")
                print("reading...")

            movie_split = read_movie_np(
                vfile, split=i, n_splits=splits, progress=progress
            )
            if progress:
                print("saving...")

            movie_h5[
                cur_row : cur_row + movie_split.shape[0], : movie_split.shape[1]
            ] = movie_split
            cur_row += movie_split.shape[0]
            
            del movie_split

    else:
        
        movie_h5[:,:,:] = read_movie_np(vfile, progress=progress)

    #### close file handle and return file name ###
    
    f.close()

    return out_file
