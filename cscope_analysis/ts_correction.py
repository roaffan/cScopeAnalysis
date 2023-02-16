import numpy as np
import pandas as pd

def run_ts_correction(mini_ts, behcam_ts, cscope_ts):

    ### correct cscope timestamp rollover

    cscope_ts["teensy_ts"] -= cscope_ts["teensy_ts"][0]
    cscope_rollover = np.where(np.diff(cscope_ts["teensy_ts"]) < 0)[0]
    for ro in cscope_rollover:
        cscope_ts["teensy_ts"][ro + 1 :] += int(np.iinfo(np.uint32).max / 1000)

    ### get the timestamps of frames from the teensy and their index
    
    cscope_frames = cscope_ts[cscope_ts["code"] == "I"]
    cscope_frames_ind = np.where(cscope_ts["code"] == "I")    

    ### remove behavior camera frames before turning on cscope LEDs
    
    behcam_ts = behcam_ts[mini_ts["ts"] > cscope_frames["teensy_ts"][0]]
    behcam_ts = behcam_ts.reset_index(drop=True)
    
    ### remove mini frames before turning on cscope LEDs
    
    mini_ts = mini_ts[mini_ts["ts"] > cscope_frames["teensy_ts"][0]]
    mini_ts = mini_ts.reset_index(drop=True)

    first_mini_ttl_ind = np.where(
        mini_ts.loc[0, "ts"] - cscope_frames["teensy_ts"] > 0
    )[0][-1]
    cscope_frames = cscope_frames[first_mini_ttl_ind:]
    cscope_frames_ind = cscope_frames_ind[first_mini_ttl_ind:]
    
    ### identify black frames (LED did not turn on)

    cscope_diff = np.diff(cscope_frames["teensy_ts"])
    cscope_events = np.where((cscope_diff > 34) | (cscope_diff < 32))[0] + 1
    cscope_events_delay = cscope_diff[cscope_events - 1]

    ### identify dropped frames (miniscope software did not save image)

    mini_diff = np.diff(mini_ts["ts"])
    mini_events = np.where((mini_diff > 34) | (mini_diff < 32))[0]
    mini_events_delay = mini_diff[mini_events]

    ### loop through and correct events

    n_dropped = 0
    n_black = 0
    dropped_frames = []
    black_frames = []

    cscope_corrected = np.copy(cscope_frames)
    cscope_corrected_ind = np.copy(cscope_frames_ind)
    mini_corrected = mini_ts.copy()
    behcam_corrected = behcam_ts.copy()

    cscope_ind = 0
    mini_ind = 0
    this_mini_event = mini_events[0]
    this_cscope_event = cscope_events[0]

    while (mini_ind < mini_events.shape[0]) or (cscope_ind < cscope_events.shape[0]):

        if this_cscope_event < this_mini_event:

            if cscope_events_delay[cscope_ind] > 34:
                
                mini_corrected.drop(
                    mini_corrected.index[this_cscope_event], inplace=True
                )
                mini_corrected = mini_corrected.reset_index(drop=True)
                mini_events[mini_events > this_cscope_event] -= 1

                this_mini_event = (
                    mini_events[mini_ind] if mini_ind < mini_events.shape[0] else np.Inf
                )

                n_black += 1
                black_frames.append(this_cscope_event)

            cscope_ind += 1
            this_cscope_event = (
                cscope_events[cscope_ind]
                if cscope_ind < cscope_events.shape[0]
                else np.Inf
            )

        else:

            if (this_mini_event < this_cscope_event) and (this_mini_event > 34):

                catch_up = 0
                next_ind = 1
                while (
                    (mini_ind + next_ind < mini_events.shape[0])
                    and (mini_events[mini_ind + next_ind] < this_cscope_event)
                    and (mini_events_delay[mini_ind + next_ind] < 33)
                ):
                    catch_up += 33 - mini_events_delay[mini_ind + next_ind]
                    next_ind += 1

                over_time = mini_events_delay[mini_ind] - 33
                n_drop = int((over_time - catch_up) / 33)

                for df in range(n_drop):

                    cscope_corrected = np.delete(cscope_corrected, this_mini_event)
                    cscope_corrected_ind = np.delete(cscope_corrected_ind, this_mini_event)
                    cscope_events[cscope_events > mini_events[mini_ind]] -= 1
                    this_cscope_event = (
                        cscope_events[cscope_ind]
                        if cscope_ind < cscope_events.shape[0]
                        else np.Inf
                    )

                    n_dropped += 1
                    dropped_frames.append(this_mini_event)

            mini_ind += 1

            this_mini_event = (
                mini_events[mini_ind] if mini_ind < mini_events.shape[0] else np.Inf
            )

    ### remove cscope timestamps after last saved mini image

    extra_cscope = cscope_corrected.shape[0] - mini_corrected.shape[0]
    cscope_corrected = cscope_corrected[:-extra_cscope]
    cscope_corrected_ind = cscope_corrected_ind[:-extra_cscope]

    ### remove blue timestamps before first green frame and after last green frame

    greens = np.where(cscope_corrected["status"] == False)[0]
    cscope_corrected = cscope_corrected[greens[0] : (greens[-1] + 1)]
    cscope_corrected_ind = cscope_corrected_ind[greens[0] : (greens[-1] + 1)]
    mini_corrected = mini_corrected[greens[0] : (greens[-1] + 1)]

    ### get index of corrected frames and bpod events

    cscope_ts['teensy_ts'][cscope_corrected_ind] = cscope_corrected['teensy_ts'] 
    keep_ind = sorted(np.append(np.where(cscope_ts['code']=='X'), cscope_corrected_ind))
    cscope_ts = cscope_ts[keep_ind]    

    
    return mini_corrected, behcam_corrected, cscope_corrected, cscope_ts
