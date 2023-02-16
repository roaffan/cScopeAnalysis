import numpy as np
import multiprocessing as mp
from multiprocessing.managers import BaseProxy
import h5py
from tqdm.auto import tqdm
from functools import partial


def hemodynamic_correct_pixel(
    green, blue, max_alpha=0, return_dff=True
):
    
    # calculate raw blue dF/F
    Ib  = blue
    Ibo = blue.mean()
    b_dff = Ib / Ibo

    # run alpha correction
    if max_alpha > 0:

        alpha_err = np.zeros(max_alpha)
        for alpha in range(max_alpha):
            g_dff = (green - alpha) / (green.mean() - alpha)
            alpha_err[alpha] = ((b_dff - g_dff) ** 2).sum()

        alpha = np.argmin(alpha_err)

    else:

        alpha = 0

    # calculate alpha-corrected green dF/F
    Ig  = green - alpha
    Igo = green.mean() - alpha
    g_dff = Ig / Igo
    
    # calculate error
    err = np.log(np.sqrt(np.sum((b_dff - g_dff) ** 2)))

    
    if return_dff:

        # calculate corrected dF/F
        b_hemo_dff = b_dff / g_dff

        return alpha, err, b_hemo_dff

    else:

        return alpha, err


def hemodynamic_correct_split(
    split,
    green_h5,
    blue_h5,
    max_alpha=0,
    n_splits=1,
    progress=True,
    write_dff=None,
    use_mc=True
):

    ### create file handle ###

    green = h5py.File(open(green_h5, "rb"), mode="r")
    blue = h5py.File(open(blue_h5, "rb"), mode="r")

    ### get rows in this split ###

    n_rows, n_cols, _ = blue["data"].shape
    split_rows = int(n_rows / n_splits)
    start_row = split * split_rows

    extra_rows = n_rows % n_splits
    if extra_rows > split:
        split_rows += 1
    start_row += min(split, extra_rows)

    ### loop through pixels ###

    alphas = np.zeros((split_rows, n_cols))
    errors = np.zeros((split_rows, n_cols))

    if (progress) and (n_splits == 1):
        row_it = tqdm(range(start_row, start_row + split_rows))
    else:
        row_it = range(start_row, start_row + split_rows)

    return_dff = True if write_dff is not None else False
    
    for r in row_it:

        for c in range(n_cols):
                   
            green_pixel = green["data"][r, c]
            blue_pixel = blue["data"][r, c]
            
            # if use_mc:
            #     green_pixel[np.where(green_pixel<1)] = np.min(green_pixel[green_pixel>0])
            #     blue_pixel[np.where(blue_pixel<1)] = np.min(blue_pixel[blue_pixel>0])
            
            res = hemodynamic_correct_pixel(
                green_pixel,
                blue_pixel,
                max_alpha,
                return_dff=return_dff,
            )
            
            if return_dff:

                alphas[r - start_row, c], errors[r - start_row, c], b_pixel_dff = res

                if isinstance(write_dff, BaseProxy):
                    write_dff.put((r - start_row, c, b_pixel_dff))
                elif isinstance(write_dff, h5py.Dataset):
                    write_dff[r - start_row, c] = b_pixel_dff
                    
            else:

                alphas[r - start_row, c], errors[r - start_row, c] = res

    return alphas, errors


def hemodynamic_correction(
    green_h5,
    blue_h5,
    max_alpha=0,
    n_processes=1,
    dff_file=None,
    progress=True,
    use_mc=True,
):

    # set up dff file if needed
    if dff_file is not None:

        # load green file to get image size ###
        green_handle = h5py.File(green_h5, mode="r")
        n_row, n_col, n_frames = green_handle["data"].shape
        green_handle.close()

        # create dff h5 file
        dff_h5 = h5py.File(dff_file, mode="w")
        dff_dset = dff_h5.create_dataset(
            "dff",
            shape=(n_row, n_col, n_frames),
            chunks=(1, 1, n_frames),
            dtype="float32",
        )

    if n_processes > 1:

        mgr = mp.Manager()
        dff_queue = mgr.Queue() if dff_file is not None else None

        pool = mp.Pool(n_processes)

        res = pool.imap(
            partial(
                hemodynamic_correct_split,
                green_h5=green_h5,
                blue_h5=blue_h5,
                max_alpha=max_alpha,
                n_splits=n_processes,
                progress=False,
                write_dff=dff_queue,
                use_mc=True,
            ),
            np.arange(n_processes),
        )

        # wait for processes to return pixel dff, write to h5 file
        if dff_file is not None:
            pixel_it = tqdm(range(n_row * n_col)) if progress else range(n_row * n_col)
            for p in pixel_it:
                row, col, dff = dff_queue.get()
                dff_dset[row, col] = dff

        # wait for processes to finish
        pool.close()
        pool.join()

        # gather alphas and errors
        alpha_proc, error_proc = zip(*res)
        alphas = alpha_proc[0]
        errors = error_proc[0]

        for i in range(1, n_processes):
            alphas = np.vstack((alphas, alpha_proc[i]))
            errors = np.vstack((errors, error_proc[i]))

    else:
        
        alphas, errors = hemodynamic_correct_split(
            0,
            green_h5,
            blue_h5,
            max_alpha=max_alpha,
            progress=progress,
            write_dff=dff_dset if dff_file is not None else None,
            use_mc=True,
        )

    if dff_file is not None:
        dff_h5.close()

    return alphas, errors
