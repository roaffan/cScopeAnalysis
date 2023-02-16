from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import cv2
import skvideo.io

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import caiman as cm
from caiman.motion_correction import MotionCorrect

from cscope_analysis.crop import cScopeCropGUI
from cscope_analysis.ts_correction import run_ts_correction
from cscope_analysis.movie import get_movie_info, movie_to_h5
from cscope_analysis.interpolation import interpolate_video
from cscope_analysis.hemodynamic import hemodynamic_correction


class cScopeRecording(object):
    def __init__(self, mini_dir=None, cscope_dir=None):

        if mini_dir is None and cscope_dir is None:
            raise Exception("Must provide at least one directory!")

        self.mini_dir = Path(mini_dir) if mini_dir is not None else None
        self.cscope_dir = (
            Path(cscope_dir)
            if cscope_dir is not None
            else self.mini_dir / "cscope-analysis"
        )

        """ find cscope data files """

        if self.mini_dir is not None:

            # miniscope software timestamps
            self.mini_ts_file = self.mini_dir / "Miniscope_V3" / "timeStamps.csv"
            self.behcam_ts_file = self.mini_dir / "BehaviorCam" / "timeStamps.csv"
            
            # cscope control box timestamps
            self.cscope_ts_file = self.mini_dir / "cScope_timestamps.npy"

            # miniscope video files (sorted)
            mini_videos = list((self.mini_dir / "Miniscope_V3").glob("*.avi"))
            mini_inds = np.array([int(mv.stem) for mv in mini_videos])
            self.mini_videos = [mini_videos[i] for i in mini_inds.argsort()]

        """ check status of analysis """

        # create cscope analysis directory create
        self.cscope_dir.mkdir(parents=True, exist_ok=True)

        # cropping parameters
        self.crop_file = self.cscope_dir / "crop.npy"
        self.crop = tuple(np.load(self.crop_file)) if self.crop_file.is_file() else None

        # corrected timestamps files
        self.corrected_ts_file = self.cscope_dir / "corrected_ts.csv"
        self.corrected_behcam_ts_file = self.cscope_dir / "corrected_behcam_ts.csv"
        self.green_ts_file = self.cscope_dir / "green_ts.csv"
        self.blue_ts_file = self.cscope_dir / "blue_ts.csv"

        # green and blue video files
        self.green_video_file = self.cscope_dir / "green.mp4"
        self.blue_video_file = self.cscope_dir / "blue.mp4"

        # green interpolated video file
        self.green_interp_file = self.cscope_dir / "green_interp.mp4"

        # motion corrected video files
        self.green_mc_file = self.cscope_dir / "green_mc.mp4"
        self.blue_mc_file = self.cscope_dir / "blue_mc.mp4"

        # hemodynamic corrected video files
        self.dff_file = self.cscope_dir / "dff.h5"
        self.hc_file = self.cscope_dir / "hemodynamic.npz"

    def set_crop(self, crop=True):

        # load first video to get image size and crop

        first_vid = cv2.VideoCapture(self.mini_videos[0].as_posix())

        if crop is True:

            cropper = cScopeCropGUI(first_vid)
            xmin, xmax, ymin, ymax = cropper.crop()
            self.crop = (xmin, xmax, ymin, ymax)

        elif (type(crop) is list) or (type(crop) is tuple):

            self.crop = tuple(crop)

        else:

            self.crop = (
                0,
                int(first_vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                0,
                int(first_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )

        np.save(self.crop_file, np.array(self.crop))
        return self.crop

    def get_mini_ts(self):

        mini_ts = pd.read_csv(self.mini_ts_file)
        mini_ts.columns = pd.Index(["frame", "ts", "buffer_idx"])
        return mini_ts
    
    def get_behcam_ts(self):

        behcam_ts = pd.read_csv(self.behcam_ts_file)
        behcam_ts.columns = pd.Index(["frame", "ts", "buffer_idx"])
        return behcam_ts

    def get_cscope_ts(self):

        return np.load(self.cscope_ts_file)

    def correct_timestamps(self, save=True, plot=False, overwrite=False):

        if (self.green_ts_file.is_file()) and (self.blue_ts_file.is_file()):

            if not overwrite:
                print(
                    "Timestamps have already been corrected. "
                    "Please use overwrite=True flag to re-run correction."
                )
                return

        ### load miniscope and cscope timestamps

        mini_ts = self.get_mini_ts()
        behcam_ts = self.get_behcam_ts()
        cscope_ts = self.get_cscope_ts()

        ### run correction

        mini_corrected, behcam_corrected, cscope_corrected, new_ts = run_ts_correction(mini_ts,behcam_ts,cscope_ts)

        ### if requested, plot corrected timestamps

        # if plot:
        #     mini_cscope_diff = mini_corrected["ts"] - cscope_corrected["teensy_ts"]
        #     plt.plot(mini_cscope_diff)
        #     plt.show()
        
        ### save new version of the control box timestamps
        new_ts = pd.DataFrame(new_ts)
        new_ts.to_csv(self.corrected_ts_file, index=False)
        
        ### split green and blue timestamps, save file

        mini_corrected["channel"] = np.where(
            cscope_corrected["status"], "blue", "green"
        )
        mini_corrected["cscope_ts"] = cscope_corrected["teensy_ts"]

        mini_corrected_green = mini_corrected.loc[cscope_corrected["status"] == False]
        mini_corrected_blue = mini_corrected.loc[cscope_corrected["status"] == True]
        mini_corrected_green.reset_index(drop=True, inplace=True)
        mini_corrected_blue.reset_index(drop=True, inplace=True)

        if save:
            mini_corrected_green.to_csv(self.green_ts_file, index=False)
            mini_corrected_blue.to_csv(self.blue_ts_file, index=False)
            behcam_corrected.to_csv(self.corrected_behcam_ts_file, index=False)
            
        return mini_corrected_green, mini_corrected_blue

    def split_videos(
        self,
        downsample=2,
        fps=30,
        crf=0,
        crop=None,
        overwrite=False,
        progress=True,
    ):

        if self.green_video_file.is_file() and self.blue_video_file.is_file():

            if not overwrite:

                print(
                    "Videos already split! Please pass overwrite=True to re-run split videos."
                )
                return

        # load first video to get image size and crop

        if crop is not None:
            self.crop = crop
        elif self.crop is None:
            warnings.warn("Crop parameters have not been set, using full image.")
            width, height, _, _ = get_movie_info(self.mini_videos[0])
            self.crop = (0, width, 0, height)

        xmin, xmax, ymin, ymax = self.crop
        im_size = (xmax - xmin + 1, ymax - ymin + 1)
        im_size_ds = (int(im_size[0] / downsample), int(im_size[1] / downsample))

        ### loop through videos and write frames

        green_ts = pd.read_csv(self.green_ts_file)
        blue_ts = pd.read_csv(self.blue_ts_file)

        green_writer = skvideo.io.FFmpegWriter(
            self.green_video_file.as_posix(),
            inputdict={"-r": f"{int(fps/2)}"},
            outputdict={
                "-vcodec": "libx264",
                "-crf": f"{crf}",
                "-preset": "veryslow",
                "-r": f"{int(fps)/2}",
            },
        )

        blue_writer = skvideo.io.FFmpegWriter(
            self.blue_video_file.as_posix(),
            inputdict={"-r": f"{int(fps/2)}"},
            outputdict={
                "-vcodec": "libx264",
                "-crf": f"{crf}",
                "-preset": "veryslow",
                "-r": f"{int(fps)/2}",
            },
        )

        frame_ind = 0

        it = (
            tqdm(range(len(self.mini_videos)))
            if progress
            else range(len(self.mini_videos))
        )

        for i in it:

            cap = cv2.VideoCapture(self.mini_videos[i].as_posix())
            ret = True

            it2 = (
                tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), leave=False)
                if progress
                else range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            )

            for j in it2:

                ret, frame = cap.read()

                if ret:

                    frame = frame[ymin:ymax, xmin:xmax]
                    frame = cv2.resize(frame, dsize=im_size_ds)
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    if frame_ind in green_ts["frame"].values:
                        green_writer.writeFrame(frame)
                    elif frame_ind in blue_ts["frame"].values:
                        blue_writer.writeFrame(frame)

                    frame_ind += 1

                else:

                    print(f"NO FRAME = {i}, {j}, {frame_ind}")

        green_writer.close()
        blue_writer.close()

    def interpolate_green(
        self,
        splits=1,
        n_processes=1,
        progress=2,
        save_h5=False,
        crf=0,
        overwrite=False,
    ):

        if self.green_interp_file.is_file():

            if not overwrite:

                print(
                    "Green videos already interpolated! Please pass overwrite=True to re-run interpolation."
                )
                return

        if (not self.green_video_file.is_file()) or (
            not self.blue_video_file.is_file()
        ):

            raise Exception("Could not find green and/or blue video file!")

        # load timestamps
        green_ts = pd.read_csv(self.green_ts_file)
        green_ts = green_ts["cscope_ts"]
        blue_ts = pd.read_csv(self.blue_ts_file)
        blue_ts = blue_ts["cscope_ts"]

        # run interpolation

        self.green_interp_file = interpolate_video(
            self.green_video_file,
            green_ts,
            blue_ts,
            new_fn=self.green_interp_file,
            splits=splits,
            n_processes=n_processes,
            progress=progress,
            save_h5=save_h5,
            crf=crf,
            overwrite=overwrite
        )

    def motion_correction(
        self, use_blue=True, parallel=False, processes=None, splits=14, overwrite=False
    ):

        if self.green_mc_file.is_file() and self.blue_mc_file.is_file():

            if not overwrite:

                print(
                    "Videos already motion corrected! Please pass overwrite=True to re-run motion correction."
                )
                return

        if (not self.green_interp_file.is_file()) or (
            not self.blue_video_file.is_file()
        ):

            raise Exception("Could not find green interpolated video and/or blue video file!")

        ### create caiman cluster for parallel motion correction ###

        _, dview, n_processes = cm.cluster.setup_cluster(
            backend="local", n_processes=processes, single_thread=not parallel
        )

        ### motion correct blue and green video ###

        print("Create motion correction object for blue and green video...")
        # video_to_mc = (
        #     self.blue_video_file.as_posix()
        #     if use_blue
        #     else self.green_interp_file.as_posix()
        # )
        # caiman_mc = MotionCorrect(video_to_mc, splits_rig=splits, 
        #                           dview=dview, border_nan=True)

        caiman_mc_blue = MotionCorrect(self.blue_video_file.as_posix(), 
                                       splits_rig=splits, 
                                       dview=dview, border_nan="copy")

        caiman_mc_green = MotionCorrect(self.green_interp_file.as_posix(),
                                        splits_rig=splits, 
                                        dview=dview, border_nan="copy")
        
        print("Calculate shifts...")
        caiman_mc_blue.motion_correct()
        caiman_mc_green.motion_correct()

        print("Apply shifts to green video...")
        green_mc_video = caiman_mc_green.apply_shifts_movie(self.green_interp_file.as_posix())

        print("Write motion corrected green video...")
        green_mc_video.save(self.green_mc_file.as_posix())

        print("Remove green video file from memory...")
        del green_mc_video

        print("Apply shifts to blue video...")
        blue_mc_video = caiman_mc_blue.apply_shifts_movie(self.blue_video_file.as_posix())

        print("Write motion corrected video...")
        blue_mc_video.save(self.blue_mc_file.as_posix())

        print("Remove blue video file from memory...")
        del blue_mc_video

        print("Stop caiman cluster...")
        if dview is not None:
            cm.stop_server(dview=dview)

        print("Done!\n")

    def hemodynamic_correction(
        self,
        max_alpha=0,
        splits=1,
        n_processes=1,
        use_mc=True,
        progress=True,
        overwrite_h5=False,
        overwrite_hc=False,
        save_hc=True,
        save_h5=False,
    ):

        if self.hc_file.is_file() or self.dff_file.is_file():

            if not overwrite_hc:

                print(
                    "Hemodynamic correction already performed! Please pass overwrite=True to re-run hemodynamic correction."
                )
                return

        if use_mc:

            if not self.green_mc_file.is_file() and self.blue_mc_file.is_file():

                print(
                    "Motion corrected videos not found. To run hemodynamic correction on raw video, please pass use_mc=False."
                )
                return

            green_file = self.green_mc_file
            blue_file = self.blue_mc_file

        else:

            green_file = self.green_interp_file
            blue_file = self.blue_video_file

        if (not green_file.is_file()) or (not blue_file.is_file()):

            raise Exception("Could not find green and/or blue video file!")

        ### check for green and blue videos as h5 files ###

        print("Saving videos as h5 files...")
        
        print("green:")
        green_h5 = movie_to_h5(
            green_file.as_posix(), splits=splits, progress=progress, overwrite=overwrite_h5
        )
        print("blue:")
        blue_h5 = movie_to_h5(
            blue_file.as_posix(), splits=splits, progress=progress, overwrite=overwrite_h5
        )

        ### Perform pixel-by-pixel hemodynamic correction ###

        print("Hemodynamic correction...")

        alphas, errors = hemodynamic_correction(
            green_h5,
            blue_h5,
            max_alpha=max_alpha,
            n_processes=n_processes,
            progress=progress,
            dff_file=self.dff_file,
            use_mc = use_mc,
        )

        ### save alpha and errors ###

        if save_hc:
            np.savez(self.hc_file, alpha=alphas, errors=errors)

        ### remove temporary (h5) files ###

        print("Check temporary files...")

        if not save_h5:

            green_h5.unlink()
            blue_h5.unlink()

        print("Done!")

        return alphas, errors
