import warnings

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import os
import h5py
import pickle
import numpy as np

import sys

sys.path.insert(0, "../")
from evaluator import *

save_data = []


def get_clusters(triggers, cluster_threshold=0.35):
    """
    Cluster a set of triggers into candidate detections.

    Arguments
    ---------
    triggers : list of triggers
        A list of triggers.  A trigger is a list of length two, where
        the first entry represents the trigger time and the second value
        represents the accompanying output value from the network.
    cluster_threshold : {float, 0.35}
        Cluster triggers together which are no more than this amount of
        time away from the boundaries of the corresponding cluster.

    Returns
    cluster_times :
        A numpy array containing the single times associated to each
        cluster.
    cluster_values :
        A numpy array containing the trigger values at the corresponing
        cluster_times.
    cluster_timevars :
        The timing certainty for each cluster. Injections must be within
        the given value for the cluster to be counted as true positive.

    """

    clusters = []
    for trigger in triggers:
        new_trigger_time = trigger[0]
        if len(clusters) == 0:
            start_new_cluster = True
        else:
            last_cluster = clusters[-1]
            last_trigger_time = last_cluster[-1][0]
            start_new_cluster = (
                new_trigger_time - last_trigger_time
            ) > cluster_threshold
        if start_new_cluster:
            clusters.append([trigger])
        else:
            last_cluster.append(trigger)

    print(
        "Clustering has resulted in {} independent triggers. Centering triggers at their maxima.".format(
            len(clusters)
        )
    )

    cluster_times = []
    cluster_values = []
    cluster_timevars = []

    # Determine maxima of clusters and the corresponding times and append them to the cluster_* lists
    for cluster in clusters:
        times = [trig[0] for trig in cluster]
        values = np.array([trig[1] for trig in cluster])
        max_index = np.argmax(values)
        cluster_times.append(times[max_index])
        cluster_values.append(values[max_index])
        cluster_timevars.append(0.3)

    cluster_times = np.array(cluster_times)
    cluster_values = np.array(cluster_values)
    cluster_timevars = np.array(cluster_timevars)

    return cluster_times, cluster_values, cluster_timevars


def run(clustering_threshold=0.0):

    print("Check clustering threshold = {}".format(clustering_threshold))
    parent_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4/"
    foutput = os.path.join(parent_dir, "testing_foutput_BEST_June_diffseed_Sept1.hdf")
    boutput = os.path.join(parent_dir, "testing_boutput_BEST_June_diffseed_Sept1.hdf")
    foreground = os.path.join(parent_dir, "foreground.hdf")
    background = os.path.join(parent_dir, "background.hdf")
    injections = os.path.join(parent_dir, "injections.hdf")

    far_scaling_factor = 2592000.0  # 30 days in seconds

    # Find indices contained in foreground
    print("Finding injections contained in data")
    padding_start, padding_end = 30, 30
    dur, idxs = find_injection_times(
        [foreground], injections, padding_start=padding_start, padding_end=padding_end
    )
    if np.sum(idxs) == 0:
        msg = "The foreground data contains no injections! "
        msg += "Probably a too small section of data was generated. "
        msg += "Please make sure to generate at least {} seconds of data. "
        msg += "Otherwise a sensitive distance cannot be calculated."
        msg = msg.format(padding_start + padding_end + 24)
        raise RuntimeError(msg)

    # Get injections optimal SNRs
    _parent_dir = "/local/scratch/igr/nnarenraju/testing_month_D4_seeded/"
    snrs_path = os.path.join(_parent_dir, "snr.hdf")
    if os.path.exists(snrs_path):
        with h5py.File(snrs_path, "r") as fp:
            snrs = fp["snr"][()]

    snrs = np.array(snrs)
    snrs = snrs[idxs]

    dataset = 4

    team_1 = {"name": "Sage"}
    team_2 = {"name": "PyCBC"}

    injparams = {}
    with h5py.File(injections, "r") as fp:
        params = list(fp.keys())
        for param in params:
            data = fp[param][()]
            injparams[param] = data[idxs]

        use_chirp_distance = "chirp_distance" in params

    other_results = "/home/nnarenraju/Research/ORChiD/results"
    other_teams = os.listdir(other_results)

    print(
        "Dataset {} comparing {} against {}".format(
            dataset, team_1["name"], team_2["name"]
        )
    )
    team_1["fgpath"] = foutput
    team_1["bgpath"] = boutput

    if team_2["name"] == "PyCBC":
        team_2["fgpath"] = os.path.join(
            other_results, "{}/ds{}/fg.hdf".format(team_2["name"], dataset)
        )
        team_2["bgpath"] = os.path.join(
            other_results, "{}/ds{}/bg.hdf".format(team_2["name"], dataset)
        )
    if team_2["name"] == "aresgw":
        team_2["fgpath"] = (
            "/home/nnarenraju/Research/ORChiD/gw-detection-deep-learning/results_best/fg.hdf"
        )
        team_2["bgpath"] = (
            "/home/nnarenraju/Research/ORChiD/gw-detection-deep-learning/results_best/bg.hdf"
        )
    if team_2["name"] in ["cWB", "MFCNN"]:
        team_2["fgpath"] = os.path.join(
            other_results, "{}/ds{}/fg.hdf".format(team_2["name"], dataset)
        )
        team_2["bgpath"] = os.path.join(
            other_results, "{}/ds{}/bg.hdf".format(team_2["name"], dataset)
        )

    for nteam in [1, 2]:
        team = locals()["team_{}".format(nteam)]
        print("\nTeam {}".format(team))
        # Read foreground events
        print(f'Reading foreground events from {team["fgpath"]}')
        fg_events = []
        with h5py.File(team["fgpath"], "r") as fp:
            fg_events.append(np.vstack([fp["time"], fp["stat"], np.array(fp["var"])]))
        team["fgevents"] = np.concatenate(fg_events, axis=-1)

        # Read background events
        print(f'Reading background events from {team["bgpath"]}')
        bg_events = []
        with h5py.File(team["bgpath"], "r") as fp:
            bg_events.append(np.vstack([fp["time"], fp["stat"], np.array(fp["var"])]))
        team["bgevents"] = np.concatenate(bg_events, axis=-1)

    print(
        "Total number of foreground triggers in unclustered Sage - Broad = {}".format(
            len(team_1["fgevents"][1])
        )
    )
    ctrigs = get_clusters(team_1["fgevents"].T, cluster_threshold=clustering_threshold)
    team_1["fgevents"] = np.stack((ctrigs))

    print(
        "Total number of background triggers in unclustered Sage - Broad = {}".format(
            len(team_1["bgevents"][1])
        )
    )
    ctrigs = get_clusters(team_1["bgevents"].T, cluster_threshold=clustering_threshold)
    team_1["bgevents"] = np.stack((ctrigs))

    """ Calculate the false-alarm rate and sensitivity of a search algorithm. """

    # Get data from fg and bg events file
    print("Team 1: {}".format(team_1["name"]))
    print("Team 2: {}".format(team_2["name"]))

    # Add SNRs into the injparams (this will automagically include it within most plots)
    chirp_distance = use_chirp_distance
    injparams["snr"] = snrs
    output_dir = "./evaluation_plots/"
    # Return data tmp var
    ret = {}

    ## COMMON ##
    # Get injection params
    injtimes = injparams["tc"]
    dist = injparams["distance"]

    # Get chirp mass from the source masses
    if chirp_distance:
        massc = mchirp(injparams["mass1"], injparams["mass2"])
    # Set duration if nothing is passed
    duration = dur

    for team in [team_1, team_2]:

        print("\nTeam {}".format(team["name"]))
        print("Sorting foreground event times")
        sidxs = team["fgevents"][0].argsort()
        fgevents = team["fgevents"].T[sidxs].T

        logging.info("Finding injection times closest to event times")
        idxs = find_closest_index(injtimes, fgevents[0])
        diff = np.abs(injtimes[idxs] - fgevents[0])

        # If the difference between the injection time and trigger is within tc variance
        # The trigger is identified as an event (there may be duplicate triggers)
        logging.info("Finding true- and false-positives")
        tpbidxs = diff <= fgevents[2]
        tpidxs = np.arange(len(fgevents[0]))[tpbidxs]
        fpbidxs = diff > fgevents[2]
        fpidxs = np.arange(len(fgevents[0]))[fpbidxs]

        tpevents = fgevents.T[tpidxs].T
        fpevents = fgevents.T[fpidxs].T

        ## Update the returns dictionary
        if team["name"] == "Sage" or team["name"] == "aresgw":
            ret["fg-events"] = fgevents
            ret["found-indices"] = np.arange(len(injtimes))[idxs]
            ret["missed-indices"] = np.setdiff1d(
                np.arange(len(injtimes)), ret["found-indices"]
            )
            ret["true-positive-event-indices"] = tpidxs
            ret["false-positive-event-indices"] = fpidxs
            ret["sorting-indices"] = sidxs
            ret["true-positive-diffs"] = diff[tpidxs]
            ret["false-positive-diffs"] = diff[fpidxs]
            ret["true-positives"] = tpevents
            ret["false-positives"] = fpevents

        # Calculate foreground FAR
        logging.info("Calculating foreground FAR")
        noise_stats_fg = fpevents[1].copy()
        noise_stats_fg.sort()
        fgfar = len(noise_stats_fg) - np.arange(len(noise_stats_fg)) - 1
        fgfar = fgfar / duration
        if team["name"] == "Sage" or team["name"] == "aresgw":
            ret["fg-far"] = fgfar

        # Calculate background FAR
        logging.info("Calculating background FAR")
        noise_stats = team["bgevents"][1].copy()
        noise_stats.sort()
        far = len(noise_stats) - np.arange(len(noise_stats)) - 1
        far = far / duration
        if team["name"] == "Sage" or team["name"] == "aresgw":
            ret["far"] = far

        # Find best true-positive for each injection
        found_injections = []
        tmpsidxs = idxs.argsort()
        sorted_idxs = idxs[tmpsidxs]
        iidxs = np.full(len(idxs), False)
        for i in tqdm(
            range(len(injtimes)), ascii=True, desc="Determining found injections"
        ):
            L = np.searchsorted(sorted_idxs, i, side="left")
            if L >= len(idxs) or sorted_idxs[L] != i:
                continue
            R = np.searchsorted(sorted_idxs, i, side="right")
            # All indices that point to the same injection
            iidxs[tmpsidxs[L:R]] = True
            # Indices of the true-positives that belong to the same injection
            eidxs = np.logical_and(iidxs[tmpsidxs[L:R]], tpbidxs[tmpsidxs[L:R]])
            if eidxs.any():
                found_injections.append([i, np.max(fgevents[1][tmpsidxs[L:R]][eidxs])])
            iidxs[tmpsidxs[L:R]] = False

        # Number of injections found within given testing data
        found_injections = np.array(found_injections).T
        print("Number of found injections = {}".format(len(found_injections[0])))

        # Calculate sensitivity
        # CARE! THIS APPLIES ONLY WHEN THE DISTRIBUTION IS CHOSEN CORRECTLY
        logging.info("Calculating sensitivity")
        sidxs = found_injections[1].argsort()  # Sort found injections
        found_injections = found_injections.T[sidxs].T

        if chirp_distance:
            found_mchirp_total = massc[found_injections[0].astype(int)]
            mchirp_max = massc.max()

        max_distance = dist.max()
        # print('Maximum distance given by injections = {}'.format(max_distance))
        vtot = (4.0 / 3.0) * np.pi * max_distance**3.0
        Ninj = len(dist)
        print("Total number of injections = {}".format(Ninj))

        # Params to calculate the sensitive volume
        if chirp_distance:
            mc_norm = mchirp_max ** (5.0 / 2.0) * len(massc)
        else:
            mc_norm = Ninj

        prefactor = vtot / mc_norm
        nfound = len(found_injections[1]) - np.searchsorted(
            found_injections[1], noise_stats, side="right"
        )

        if chirp_distance:
            # Get found chirp-mass indices for given threshold
            fidxs = np.searchsorted(found_injections[1], noise_stats, side="right")
            # Plotting the network output
            # network_output(found_injections, noise_stats, output_dir, team['name'], lower_threshold=-999)

            found_mchirp_total = np.flip(found_mchirp_total)

            # Calculate sum(found_mchirp ** (5/2))
            # with found_mchirp = found_mchirp_total[i:]
            # and i looped over fidxs
            # Code below is a vectorized form of that
            cumsum = np.flip(np.cumsum(found_mchirp_total ** (5.0 / 2.0)))
            cumsum = np.concatenate([cumsum, np.zeros(1)])
            mc_sum = cumsum[fidxs]
            Ninj = np.sum((mchirp_max / massc) ** (5.0 / 2.0))

            cumsumsq = np.flip(np.cumsum(found_mchirp_total**5))
            cumsumsq = np.concatenate([cumsumsq, np.zeros(1)])
            sample_variance_prefactor = cumsumsq[fidxs]
            sample_variance = (
                sample_variance_prefactor / Ninj - (mc_sum / Ninj) ** 2
            )  # noqa: E127
        else:
            mc_sum = nfound
            sample_variance = nfound / Ninj - (nfound / Ninj) ** 2

        vol = prefactor * mc_sum
        vol_err = prefactor * (Ninj * sample_variance) ** 0.5
        rad = (3 * vol / (4 * np.pi)) ** (1.0 / 3.0)
        print(
            "Radius or sensitive distance as calculated from the volume obtained ({})".format(
                team["name"]
            )
        )
        print("min rad = {}, max rad = {}".format(min(rad), max(rad)))

        if team["name"] == "Sage" or team["name"] == "aresgw":
            ret["sensitive-volume"] = vol
            ret["sensitive-distance"] = rad
            ret["sensitive-volume-error"] = vol_err
            ret["sensitive-fraction"] = nfound / Ninj

        if team["name"] == "aresgw":
            with h5py.File("./evaluation_plots/evaluation_aresgw.hdf", "w") as fp:
                for key, val in ret.items():
                    fp.create_dataset(key, data=np.array(val))

        if team["name"] == "PyCBC":  # or team['name'] == "aresgw":
            ret["sensitive-distance-pycbc"] = rad
            ret["far-pycbc"] = far

        # Update plotting params for each group
        team["found_idx"] = found_injections[0].astype(int)
        team["found_stats"] = found_injections[1]
        # Add all found injparams to to plotting dict
        team["params"] = list(injparams.keys())
        team.update(injparams)
        # The values given are indices and have to be 1 less than the number of FA per month req.
        team["far_thresholds"] = noise_stats[::-1][[0, 3, 29, 99, 999]]
        team["noise_stats"] = noise_stats[::-1]
        if team["name"] == "Sage" and False:
            with open("noise_stats.pickle", "wb") as handle:
                pickle.dump(
                    team["noise_stats"], handle, protocol=pickle.HIGHEST_PROTOCOL
                )
        team["sens_dist"] = rad
        team["sens_frac"] = nfound / Ninj

    far_scaling_factor = 2592000.0
    far = ret["far"]
    sens = ret["sensitive-distance"]
    sidxs = ret["far"].argsort()
    far = far[sidxs][1:] * far_scaling_factor
    sens = sens[sidxs][1:]
    check_1 = sens[:1000]

    save_data.append([check_1[0], check_1[9], check_1[99], check_1[999]])


if __name__ == "__main__":

    for foo in np.arange(0.0, 10.001, 0.01):
        run(clustering_threshold=foo)

    print(save_data)
    save_data = np.array(save_data)
    np.save("./cluster_data.npy", save_data)
