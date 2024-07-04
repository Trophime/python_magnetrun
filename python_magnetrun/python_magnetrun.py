"""Main module."""

import os
import traceback

from .MagnetRun import MagnetRun
from .processing.smoothers import savgol
from scipy.signal import find_peaks

from tabulate import tabulate

# import logging

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available


def plot_bkpts(
    file: str,
    channel: str,
    symbol: str,
    unit: str,
    ts: pd.DataFrame,
    smoothed: np.ndarray,
    smoothed_der1: np.ndarray,
    smoothed_der2: np.ndarray,
    quantiles_der: float,
    peaks: np.ndarray,
    ignore_peaks: list[int],
    anomalies: list[int],
    save: bool = False,
):
    """_summary_

    :param file: _description_
    :type file: str
    :param channel: _description_
    :type channel: str
    :param symbol: _description_
    :type symbol: str
    :param unit: _description_
    :type unit: str
    :param ts: _description_
    :type ts: pd.tseries
    :param smoothed: _description_
    :type smoothed: np.ndarray
    :param smoothed_der1: _description_
    :type smoothed_der1: np.ndarray
    :param smoothed_der2: _description_
    :type smoothed_der2: np.ndarray
    :param quantiles_der: _description_
    :type quantiles_der: float
    :param peaks: _description_
    :type peaks: np.ndarray
    :param ignore_peaks: _description_
    :type ignore_peaks: list[int]
    :param anomalies: _description_
    :type anomalies: list[int]
    :param save: _description_, defaults to False
    :type save: bool, optional
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 1)

    ax0 = plt.subplot(gs[0])
    ax0.plot(ts.to_numpy(), label=key, color="blue", marker="o", linestyle="None")
    ax0.plot(smoothed, label="smoothed", color="red")
    ax0.legend()
    ax0.grid()
    # ax0.set_xlabel('t [s]')
    ax0.set_ylabel(f"{symbol} [{unit:~P}]")
    ax0.set_title(f"{file}: {key}")

    ax1 = plt.subplot(gs[1])
    ax1.plot(smoothed_der2, label=key, color="red")
    ax1.legend()
    ax1.grid()
    # ax1.set_xlabel('t [s]')
    ax1.set_title(f"Savgo filter [2nd order der]: ({level}\%: {quantiles_der:.3e})")

    ax2 = plt.subplot(gs[2])
    std_ts = ts.rolling(window=args.window).std()
    ax2.plot(std_ts.to_numpy(), label="rolling std", color="blue")
    ax2.legend()
    ax2.grid()
    ax2.set_xlabel("t [s]")
    ax2.set_title("Rolling std")

    if peaks.shape[0]:
        ax0.plot(peaks, smoothed[peaks], "go", label="peaks")
        ax0.legend()

        ax1.plot(peaks, smoothed_der2[peaks], "go", label="peaks")
        ax1.legend()

    if ignore_peaks:
        ax0.plot(ignore_peaks, smoothed[ignore_peaks], "yo", label="ignore peaks")
        ax0.legend()

        ax1.plot(ignore_peaks, smoothed_der2[ignore_peaks], "yo", label="ignore peaks")
        ax1.legend()

    if anomalies:
        ax0.plot(anomalies, smoothed[anomalies], "ro", label="anomalies")
        ax0.legend()

        ax1.plot(anomalies, smoothed_der2[anomalies], "ro", label="anomalies")
        ax1.legend()

    if save:
        plt.savefig(
            f'{file.replace(f_extension,"")}-{channel}-detect_bkpts.png', dpi=300
        )
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="+", help="enter input file")
    parser.add_argument(
        "--site", help="specify a site (ex. M8, M9,...)", default="M9"
    )  # use housing instead
    parser.add_argument("--insert", help="specify an insert", default="notdefined")
    parser.add_argument("--debug", help="acticate debug", action="store_true")

    subparsers = parser.add_subparsers(
        title="commands", dest="command", help="sub-command help"
    )
    parser_info = subparsers.add_parser("info", help="info help")
    parser_select = subparsers.add_parser("select", help="select help")
    parser_stats = subparsers.add_parser("stats", help="stats help")
    parser_plot = subparsers.add_parser("plot", help="select help")

    # add info subcommand
    parser_info.add_argument("--list", help="list key in csv", action="store_true")
    parser_info.add_argument("--convert", help="save to csv", action="store_true")

    # add plot subcommand
    parser_plot.add_argument(
        "--normalize", help="normalize data before plot", action="store_true"
    )
    parser_plot.add_argument(
        "--vs_time",
        help='select key(s) to plot (ex. "Field [Ucoil1]")',
        nargs="+",
        action="append",
    )
    parser_plot.add_argument(
        "--key_vs_key",
        help='select pair(s) of keys to plot (ex. "Field-Icoil1")',
        nargs="+",
        action="append",
    )
    parser_plot.add_argument(
        "--save", help="save graphs (png format)", action="store_true"
    )

    # extract subcommand
    parser_select.add_argument(
        "--output_time", nargs="+", help="output key(s) for time"
    )
    parser_select.add_argument(
        "--output_timerange",
        help="set time range to extract (start;end)",
        action="append",
    )
    parser_select.add_argument(
        "--output_key",
        nargs="+",
        help="output key(s) for time",
        action="append",
    )
    parser_select.add_argument(
        "--extract_pairkeys",
        nargs="+",
        help="dump key(s) to file",
        action="append",
    )
    parser_select.add_argument(
        "--convert", help="convert file to csv", action="store_true"
    )

    # add stats subcommand
    parser_stats.add_argument(
        "--detect_bkpts", help="find breaking points", action="store_true"
    )
    parser_stats.add_argument("--plateau", help="find plateau", action="store_true")
    parser_stats.add_argument(
        "--save", help="save graphs (png format)", action="store_true"
    )
    parser_stats.add_argument(
        "--show", help="display graphs (require X11)", action="store_true"
    )
    parser_stats.add_argument(
        "--keys",
        help="select key(s) to perform selected stats",
        nargs="+",
    )
    parser_stats.add_argument(
        "--threshold",
        help="specify threshold for regime detection",
        type=float,
        default=1.0e-3,
    )
    parser_stats.add_argument(
        "--bthreshold",
        help="specify b threshold for regime detection",
        type=float,
        default=1.0e-3,
    )
    parser_stats.add_argument(
        "--dthreshold",
        help="specify duration threshold for regime detection",
        type=float,
        default=10,
    )
    parser_stats.add_argument(
        "--window", help="stopping criteria for nlopt", type=int, default=10
    )
    parser_stats.add_argument("--level", help="select level", type=int, default=90)
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    # load df pandas from input_file
    # check extension
    supported_formats = [".txt", ".tdms", ".csv"]

    inputs = {}
    extensions = {}
    for i, file in enumerate(args.input_file):
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in extensions:
            extensions[f_extension] = [i]
        else:
            extensions[f_extension].append(i)
    # print(f"extensions: {extensions}")

    for file in args.input_file:
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in supported_formats:
            raise RuntimeError(
                f"so far file with extension in {supported_formats} are implemented"
            )

        filename = os.path.basename(file)
        result = filename.startswith("M")
        insert = "tututu"
        if result:
            try:
                index = filename.index("_")
                site = filename[0:index]
                print(f"site detected: {site}")
            except Exception:
                print("no site detected - use args.site argument instead")
                pass

        match f_extension:
            case ".txt":
                mrun = MagnetRun.fromtxt(site, insert, file)
            case ".tdms":
                mrun = MagnetRun.fromtdms(site, insert, file)
            case ".csv":
                mrun = MagnetRun.fromcsv(site, insert, file)
            case _:
                raise RuntimeError(
                    f"so far file with extension in {supported_formats} are implemented"
                )

        inputs[file] = {"data": mrun}

        if args.command == "info":
            mdata = mrun.getMData()
            mdata.info()
            if args.list:
                print(f"{file}: valid keys")
                for key in mrun.getKeys():
                    print("\t", key)

            if args.convert:
                data = mdata.Data
                if mdata.Type == 0:
                    csvfile = file.replace(f_extension, ".csv")
                    data.to_csv(csvfile, sep=str("\t"), index=True, header=True)
                elif mdata.Type == 1:
                    for key, df in data.items():
                        print(f"convert: key={key}", flush=True)
                        csvfile = file.replace(f_extension, f"-{key}.csv")
                        df.to_csv(csvfile, sep=str("\t"), index=True, header=True)

    # perform operations defined by options
    if args.command == "plot":
        if args.vs_time:
            assert (
                len(args.vs_time) == len(extensions)
            ), f"expected {len(extensions)} vs_time arguments - got {len(args.vs_time)} "

            my_ax = plt.gca()

            items = args.vs_time
            # print(f"items={items}", flush=True)
            title = ""
            for file in args.input_file:
                title = os.path.basename(file)
                f_extension = os.path.splitext(file)[-1]
                plot_args = items[extensions[f_extension][0]]
                mrun: MagnetRun = inputs[file]["data"]
                mdata = mrun.getMData()
                for key in plot_args:
                    # print(f"plot key={key}, type={type(key)}", flush=True)
                    mdata.plotData(x="t", y=key, ax=my_ax, normalize=args.normalize)

                if args.normalize:
                    plt.ylabel("")

            plt.title(title)
            if not args.save:
                plt.show()
            else:
                imagefile = "image"
                print(f"saveto: {imagefile}_vs_time.png", flush=True)
                plt.savefig(f"{imagefile}_vs_time.png", dpi=300)
            plt.close()

        if args.key_vs_key:
            assert (
                len(args.key_vs_key) == len(extensions)
            ), f"expected {len(extensions)} key_vs_key arguments - got {len(args.key_vs_key)} "

            my_ax = plt.gca()

            # split pairs in key1, key2
            print(f"key_vs_key={args.key_vs_key}")
            pairs = args.key_vs_key
            for file in args.input_file:
                f_extension = os.path.splitext(file)[-1]
                plot_args = pairs[extensions[f_extension][0]]
                mrun: MagnetRun = inputs[file]["data"]
                mdata = mrun.getMData()

                for pair in plot_args:
                    print(f"pair={pair}")
                    # print("pair=", pair, " type=", type(pair))
                    items = pair.split("-")
                    if len(items) != 2:
                        raise RuntimeError(f"invalid pair of keys:{pair}")
                    key1 = items[0]
                    key2 = items[1]
                    mdata.plotData(x=key1, y=key2, ax=my_ax)

            if not args.save:
                plt.show()
            else:
                imagefile = "image"
                print(f"saveto: {imagefile}_key_vs_key.png", flush=True)
            plt.savefig(f"{imagefile}_key_vs_key.png", dpi=300)
            plt.close()

    if args.command == "select":
        # plot_args = items[extensions[f_extension][0]]
        if args.output_time:
            assert (
                len(args.output_time) == len(extensions)
            ), f"expected {len(extensions)} output_time arguments - got {len(args.output_time)} "

            times = args.output_time.split(";")
            print(f"Select data at {times}")
            for file in inputs:
                f_extension = os.path.splitext(file)[-1]
                mrun: MagnetRun = inputs[file]["data"]
                mdata = mrun.getMData()
                select_args = times[extensions[f_extension][0]]
                select_args_str = "_at"
                for item in select_args:
                    select_args_str += f"-{item:.3f}s"

                if mdata.Type == 0:
                    data = mdata.Data
                    df = data[data["timestamp"].isin(times)]
                    file_name = file.replace(f_extension, "")
                    file_name = file_name + select_args_str + ".csv"
                    df.to_csv()

                elif mdata.Type == 1:
                    data = mdata.Data
                    for group in data:
                        df = data[data.index.isin(times)]
                        file_name = file.replace(f_extension, f"-{group}")
                        file_name = file_name + select_args_str + ".csv"
                        df.to_csv()

        if args.output_timerange:
            assert (
                len(args.output_timerange) == len(extensions)
            ), f"expected {len(extensions)} output_timerange arguments - got {len(args.output_timerange)} "

            timerange = args.output_timerange.split(";")
            for file in inputs:
                f_extension = os.path.splitext(file)[-1]
                mrun: MagnetRun = inputs[file]["data"]
                mdata = mrun.getMData()
                select_args = args.output_timerange[extensions[f_extension][0]]
                for item in select_args:
                    timerange = item.split(";")

                    file_name = file.replace(f_extension, "")
                    file_name = (
                        file_name + "_from" + str(timerange[0].replace(":", "-"))
                    )
                    file_name = (
                        file_name + "_to" + str(timerange[1].replace(":", "-")) + ".csv"
                    )

                    if mdata.Type == 0:
                        selected_df = mdata.extractTimeData(timerange)
                        if selected_df is not None:
                            selected_df.to_csv(
                                file_name, sep=str("\t"), index=False, header=True
                            )
                    elif mdata.Type == 1:
                        for group in mdata.Data:
                            selected_df = mdata.extractTimeData(timerange, group)
                            if selected_df is not None:
                                selected_df.to_csv(
                                    file_name, sep=str("\t"), index=False, header=True
                                )

        if args.output_key:
            assert (
                len(args.output_key) == len(extensions)
            ), f"expected {len(extensions)} output_key arguments - got {len(args.output_key)} "

            for file in inputs:
                f_extension = os.path.splitext(file)[-1]
                mrun: MagnetRun = inputs[file]["data"]
                mdata = mrun.getMData()
                select_args = args.output_key[extensions[f_extension][0]]
                for item in select_args:
                    keys = item.split(";")
                    keys.insert(0, "Time")

                    file_name = file.replace(".txt", "")
                    for key in keys:
                        if key != "Time":
                            file_name = file_name + "_" + key
                    file_name = file_name + "_vs_Time.csv"

                    selected_df = mdata.extractData(keys)
                    if selected_df is not None:
                        selected_df.to_csv(
                            file_name, sep=str("\t"), index=False, header=True
                        )

        if args.extract_pairkeys:
            assert (
                len(args.extract_pairkeys) == len(extensions)
            ), f"expected {len(extensions)} extract_pairkeys arguments - got {len(args.extract_pairkeys)} "
            for file in inputs:
                f_extension = os.path.splitext(file)[-1]
                mrun: MagnetRun = inputs[file]["data"]
                mdata = mrun.getMData()
                select_args = args.extract_pairkeys[extensions[f_extension][0]]
                for item in select_args:
                    pairs = item.split(";")
                    for pair in pairs:
                        items = pair.split("-")
                        if len(items) != 2:
                            raise RuntimeError(f"invalid pair of keys: {pair}")
                        key1 = items[0]
                        key2 = items[1]
                        if mdata is not None:
                            newdf = mdata.extractData([key1, key2])
                            if newdf is not None:
                                # Remove line with I=0
                                newdf = newdf[newdf[key1] != 0]
                                newdf = newdf[newdf[key2] != 0]

                                file_name = str(pair) + ".csv"
                                newdf.to_csv(
                                    file_name, sep=str("\t"), index=False, header=False
                                )

        if args.convert:
            for file in inputs:
                mrun: MagnetRun = inputs[file]["data"]
                mdata = mrun.getMData()

                extension = os.path.splitext(file)[-1]
                file_name = args.input_file.replace(extension, ".csv")
                if mdata.Type == 0:
                    mdata.to_csv(file_name, sep=str("\t"), index=False, header=True)

    if args.command == "stats":
        from .processing import stats

        if args.plateau:
            from .utils.plateaux import nplateaus

        print("Stats:")
        for file in inputs:
            print(file)
            extension = os.path.splitext(file)[-1]
            mrun: MagnetRun = inputs[file]["data"]
            mdata = mrun.getMData()

            if not args.plateau and not args.detect_bkpts:
                stats.stats(mdata)

            try:
                if args.keys:
                    for key in args.keys:
                        if mdata.Type == 0:
                            print(f"pupitre: stats for {key}", flush=True)
                            (symbol, unit) = mdata.getUnitKey(key)

                            period = 1
                            num_points_threshold = int(args.dthreshold / period)
                        elif mdata.Type == 1:
                            print(f"pigbrother: stats for {key}", flush=True)
                            (symbol, unit) = mdata.getUnitKey(key)

                            # compute num_points_threshold from dthresold
                            (group, channel) = key.split("/")
                            period = mdata.Groups[group][channel]["wf_increment"]
                            num_points_threshold = int(args.dthreshold / period)

                        if args.plateau:
                            pdata = nplateaus(
                                mdata,
                                xField=("t", "t", "s"),
                                yField=(key, symbol, unit),
                                threshold=args.threshold,
                                num_points_threshold=num_points_threshold,
                                save=args.save,
                                show=args.show,
                                verbose=False,
                            )
                            print(f"display plateaus for {key}")
                            df_plateaux = pd.DataFrame()
                            for entry in ["start", "end", "value"]:
                                df_plateaux[entry] = [
                                    plateau[entry] for plateau in pdata
                                ]

                            # rename column value using symbol and unit
                            print(
                                tabulate(
                                    df_plateaux,
                                    headers="keys",
                                    tablefmt="psql",
                                    showindex=False,
                                )
                            )

                        if args.detect_bkpts:
                            ts = None
                            if mdata.Type == 0:
                                ts = mdata.Data[key]
                                freq = 1
                                print(f"{key}: freq={freq} Hz", flush=True)
                            elif mdata.Type == 1:
                                ts = mdata.Data[group][channel]
                                freq = 1 / mdata.Groups[group][channel]["wf_increment"]
                                print(f"{group}/{channel}: freq={freq} Hz", flush=True)

                            smoothed = savgol(
                                y=ts.to_numpy(),
                                window=args.window,
                                polyorder=3,
                                deriv=0,
                            )
                            print(f"{file}: stats for smoothed")
                            print(f"min: {abs(smoothed).min()}")
                            print(f"mean: {abs(smoothed).mean()}")
                            print(f"max: {abs(smoothed).max()}")
                            print(f"std: {abs(smoothed).std()}")
                            quantiles = {}
                            for level in range(5, 100, 5):
                                quantiles[str(level)] = np.quantile(
                                    abs(smoothed), level / 100.0
                                )

                            level = args.level
                            max_level_50 = (
                                abs(1 - abs(smoothed).max() / quantiles["50"]) * 100.0
                            )
                            max_level_75 = (
                                abs(1 - abs(smoothed).max() / quantiles["75"]) * 100.0
                            )
                            print(
                                f"max_level_50={max_level_50}, max_level_75={max_level_75}"
                            )
                            if max_level_75 >= 5000:
                                print(f"overwrite level: {level} -> 40")
                                level = 40
                            if max_level_75 >= 1000:
                                print(f"overwrite level: {level} -> 80")
                                level = 80
                            if max_level_75 <= 500:
                                print(f"overwrite level: {level} -> 95")
                                level = 95
                            if max_level_75 <= 60:
                                print(f"overwrite level: {level} -> 96")
                                level = 96
                            if max_level_75 <= 20:
                                print(f"overwrite level: {level} -> 97")
                                level = 97
                            if max_level_75 <= 10:
                                print(f"overwrite level: {level} -> 98")
                                level = 98
                            if max_level_75 <= 0.1:
                                print(f"overwrite level: {level} -> 99")
                                level = 99
                            if max_level_75 <= 0.02:
                                print(f"overwrite level: {level} -> 99.7")
                                level = 99.7
                            # print(f'{file}: {max_level}%', flush=True)

                            smoothed_der1 = savgol(
                                y=ts.to_numpy(),
                                window=args.window,
                                polyorder=3,
                                deriv=1,
                            )
                            smoothed_der2 = savgol(
                                y=ts.to_numpy(),
                                window=args.window,
                                polyorder=3,
                                deriv=2,
                            )
                            print(f"{file}: stats for smoother 2nd order derivate")
                            print(f"min: {abs(smoothed_der2).min()}")
                            print(f"mean: {abs(smoothed_der2).mean()}")
                            print(f"max: {abs(smoothed_der2).max()}")
                            print(f"std: {abs(smoothed_der2).std()}")
                            quantiles_der = np.quantile(
                                abs(smoothed_der2), level / 100.0
                            )

                            # find peak of der2
                            peaks, peaks_properties = find_peaks(
                                abs(smoothed_der2), height=quantiles_der
                            )

                            # get peaks where std is above a giventhresold
                            # filtered_std_df = std_ts.gt(10)
                            ignore_peaks = []
                            """
                            for peak in peaks:
                                # print(f'peak={peak}', flush=True)
                                num = peak-1  # args.window
                                before = smoothed_der1[num]
                                num = peak+1 #args.window
                                after = smoothed_der1[num]
                                diff = abs(before-after)
                                # print(f'{peak}: before={before} after={after} diff={diff}', end="")
                                if diff <= 1:
                                    # print(' **')
                                    ignore_peaks.append(peak)
                                print(flush=True)
                            """
                            print(
                                f"{channel}: peaks={peaks.shape[0]}, ignore_peaks={len(ignore_peaks)}"
                            )

                            plot_bkpts(
                                file,
                                channel,
                                symbol,
                                unit,
                                ts,
                                smoothed,
                                smoothed_der1,
                                smoothed_der2,
                                quantiles_der,
                                peaks,
                                ignore_peaks,
                                [],
                                args.save,
                            )

                            if mdata.Type == 1:
                                # select key from GR1 or GR2
                                selected = [
                                    t
                                    for t in mdata.Keys
                                    if t.startswith("Tensions_Aimant/Interne")
                                ]
                                print(f"selected: {selected}")
                                for key in selected:
                                    (symbol, unit) = mdata.getUnitKey(key)

                                    (group, channel) = key.split("/")
                                    period = mdata.Groups[group][channel][
                                        "wf_increment"
                                    ]
                                    num_points_threshold = int(args.dthreshold / period)

                                    ts = mdata.Data[group][channel]
                                    smoothed = savgol(
                                        y=ts.to_numpy(),
                                        window=args.window,
                                        polyorder=3,
                                        deriv=0,
                                    )
                                    smoothed_der1 = savgol(
                                        y=ts.to_numpy(),
                                        window=args.window,
                                        polyorder=3,
                                        deriv=1,
                                    )
                                    smoothed_der2 = savgol(
                                        y=ts.to_numpy(),
                                        window=args.window,
                                        polyorder=3,
                                        deriv=2,
                                    )
                                    quantiles_der = np.quantile(
                                        abs(smoothed_der2), level / 100.0
                                    )
                                    cpeaks, cpeaks_properties = find_peaks(
                                        abs(smoothed_der2), height=quantiles_der
                                    )

                                    if cpeaks.shape[0] != peaks.shape[0]:
                                        # print(f'{channel}: peaks={cpeaks.shape[0]}')
                                        # print(f'peaks: {peaks}')
                                        # print(f'cpeaks: {cpeaks}')
                                        isin = np.isin(cpeaks, peaks)
                                        anomalies = []
                                        real_anomalies = []
                                        for i, item in enumerate(isin):
                                            if not item:
                                                first = np.isin([cpeaks[i] - 1], peaks)
                                                last = np.isin([cpeaks[i] + 1], peaks)
                                                # print(cpeaks[i], first, last, item)
                                                if not first[0] and not last[0]:
                                                    anomalies.append(cpeaks[i])

                                                    # calculate the difference array
                                                    difference_array = np.absolute(
                                                        peaks - cpeaks[i]
                                                    )

                                                    # find the index of minimum element from the array
                                                    index = difference_array.argmin()
                                                    msg = f"{i}: closest values in peaks={peaks[index]}, cpeaks[{i}]={cpeaks[i]}"

                                                    if (
                                                        abs(peaks[index] - cpeaks[i])
                                                        >= args.window
                                                    ):
                                                        msg += " **"
                                                        real_anomalies.append(cpeaks[i])
                                                    print(f"{msg}")

                                        print(
                                            f"anomalies: {len(anomalies)} - likely {len(real_anomalies)}"
                                        )
                                        if real_anomalies:
                                            # if args.verbose:
                                            #     print(f"anomalies: {anomalies}")
                                            plot_bkpts(
                                                file,
                                                channel,
                                                symbol,
                                                unit,
                                                ts,
                                                smoothed,
                                                smoothed_der1,
                                                smoothed_der2,
                                                quantiles_der,
                                                cpeaks,
                                                [],
                                                real_anomalies,
                                                args.save,
                                            )

            except Exception:
                print(traceback.format_exc())
                pass
