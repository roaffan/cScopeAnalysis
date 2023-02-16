import numpy as np
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
from pathlib import Path
import h5py
import multiprocessing as mp
from functools import partial
import cv2
import skvideo


from cscope_analysis.movie import get_movie_info, read_movie_np


def interpolate_pixel(pixel, ts, new_ts, dtype=np.uint8):
    pixel_fxn = interp1d(ts, pixel, bounds_error=False)
    return pixel_fxn(new_ts).astype(dtype)


def interpolate_chunk(vdata, ts, new_ts, dtype=np.uint8, progress=True):

    chunk_interp = np.zeros(
        (vdata.shape[0], vdata.shape[1], new_ts.shape[0]), dtype=dtype
    )

    row_it = (
        tqdm(range(vdata.shape[0]), leave=False) if progress else range(vdata.shape[0])
    )

    for r in row_it:
        for c in range(vdata.shape[1]):
            chunk_interp[r, c] = interpolate_pixel(vdata[r, c], ts, new_ts, dtype=dtype)

    return chunk_interp


def interpolate_video(
    vfile,
    ts,
    new_ts,
    new_fn=None,
    splits=1,
    n_processes=1,
    progress=2,
    save_h5=False,
    crf=0,
    overwrite=False,
):

    ### set up files / check for interpolated video ###

    vfile = Path(vfile)
    interp_h5_file = vfile.parents[0] / f"{vfile.stem}_interp.h5"
    interp_vfile = (
        interp_h5_file.with_suffix(".mp4") if new_fn is None else Path(new_fn)
    )

    if interp_vfile.is_file():
        if not overwrite:
            print(
                f"interpolated video = {interp_vfile.as_posix()} already exists! Please pass overwrite=False to overwrite this file."
            )
            return interp_vfile

    ### interpolation ###

    n_cols, n_rows, _, _ = get_movie_info(vfile)

    if (not interp_h5_file.is_file()) or (overwrite):

        interp_h5 = h5py.File(interp_h5_file, mode="w")
        interp_dset = interp_h5.create_dataset(
            "data",
            shape=(n_rows, n_cols, new_ts.shape[0]),
            chunks=(1, n_cols, 1),
            dtype="uint8",
        )

        # read movie splits, calculate interpolation, save to h5 in chunks

        if progress > 0:
            print(
                "read movie, calculate inteprolation, and write to h5 file (in chunks)..."
            )

        cur_row = 0

        for s in range(splits):

            if progress > 0:
                print(f"split = {s+1} / {splits}; read movie split")

            movie_split = read_movie_np(
                vfile.as_posix(), split=s, n_splits=splits, progress=(progress > 1)
            )

            if progress > 0:
                print(f"split = {s+1} / {splits}; calculate interpolation")

            if n_processes > 1:

                pool = mp.Pool(n_processes)
                split_interp = pool.imap(
                    partial(
                        interpolate_chunk,
                        ts=ts,
                        new_ts=new_ts,
                        progress=False,
                    ),
                    np.array_split(movie_split, n_processes),
                )
                pool.close()
                pool.join()

                split_interp = np.vstack(list(split_interp))

            else:

                split_interp = interpolate_chunk(
                    movie_split, ts, new_ts, progress=(progress > 1)
                )

            if progress > 0:
                print(f"split = {s+1} / {splits}; write to h5 file")

            write_it = (
                tqdm(range(split_interp.shape[0]), leave=False)
                if progress > 1
                else range(split_interp.shape[0])
            )
            for w in write_it:
                interp_dset[cur_row + w] = split_interp[w]
            cur_row += split_interp.shape[0]
    else:

        if progress > 0:
            print("interpolated h5 file already exists!")

        interp_h5 = h5py.File(interp_h5_file, mode="r")
        interp_dset = interp_h5["data"]
        interp_h5.close()


    ### read individual frames and write to mp4 ###

    if progress > 0:
        print("writing video...")

    fps = np.round(1000 / np.mean(np.diff(new_ts))).astype(int)

    vwriter = skvideo.io.FFmpegWriter(
        interp_vfile.as_posix(),
        inputdict={"-r": f"{fps}"},
        outputdict={
            "-vcodec": "libx264",
            "-crf": f"{crf}",
            "-preset": "veryslow",
            "-r": f"{fps}",
        },
    )

    frame_it = (
        tqdm(range(interp_dset.shape[2]))
        if progress > 1
        else range(interp_dset.shape[2])
    )
    for f in frame_it:
        frame = cv2.cvtColor(interp_dset[:, :, f], cv2.COLOR_GRAY2BGR)
        vwriter.writeFrame(frame)

    vwriter.close()

    if not save_h5:
        interp_h5.close()
        interp_h5_file.unlink()

    return interp_vfile
