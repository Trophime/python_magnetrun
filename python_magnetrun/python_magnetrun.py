"""Main module."""

import os
import sys
import datetime
import magnetdata
import file_utils
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available

class MagnetRun:
    """
    Magnet Run

    Site: name of the site
    Insert: list of the MagnetIDs composing the insert
    MagnetData: pandas dataframe or tdms file
    """

    def __init__(self, site="M9", insert="", filename="tutu"):
        self.Site = site
        self.MagnetData = magnetdata.MagnetData(filename)
        with open(filename, 'r') as f:
            f_extension=os.path.splitext(filename)[-1]
            if f_extension == ".txt":
                self.Insert=f.readline().split()[-1]

                if "Date" in self.Keys and "Time" in self.MagnetDataKeys:
                    tformat="%Y.%m.%d %H:%M:%S"
                    start_date=self.Data["Date"].iloc[0]
                    start_time=self.Data["Time"].iloc[0]
                    end_time=self.Data["Time"].iloc[-1]
                    print ("start_time=", start_time, "start_date=", start_date)

    def __repr__(self):
        return "%s(Site=%r, Insert=%r, MagnetData=%r)" % \
            (self.__class__.__name__,
             self.Site,
             self.Insert,
             self.MagnetData)

    def getSite(self):
        """returns Site"""
        return self.Site

    def setSite(self, site):
        """set Site"""
        self.Site = site

    def getType(self):
        """returns Data Type"""
        return self.MagnetData.Type

    def getMData(self):
        """retunr Data"""
        return self.MagnetData

    def getData(self):
        """retunr Data"""
        return self.MagnetData.getData()

    def getKeys(self):
        """return list of Data keys"""
        return self.MagnetData.Keys


    def stats(self):
        """compute stats fro the actual run"""
        thresold = 1.e-3 #3.e-5
        
        if self.MagnetData.Type == 0:

            if self.Site == "M9":
                self.MagnetData.addData("IH", "IH = Idcct1 + Idcct2")
                self.MagnetData.addData("IB", "IB = Idcct3 + Idcct4")
            elif self.Site in ["M8", "M10"]:
                self.MagnetData.addData("IH", "IH = Idcct3 + Idcct4")
                self.MagnetData.addData("IB", "IB = Idcct1 + Idcct2")
            else:
                raise Exception("stats: unknown site %s " % self.Site)
            
            # Try to detect plateaus
            # Compute "normalized" Gradient(I)
            #  0 -> plateau
            #  1 -> up
            # -1 -> down
            ax = plt.gca()
            for key in ["IH", "IB"]:
                I1_max = self.MagnetData.getData()[key].max()

                regime = self.MagnetData.getData()[key].to_numpy()
                df_ = pd.DataFrame(regime)
                df_['regime']=pd.Series(regime)

                df_[0] /= I1_max
                diff = np.diff(regime)
                df_['diff']=pd.Series(diff)

                ndiff = np.where(abs(diff) >= thresold, diff, 0)
                df_['ndiff']=pd.Series(ndiff)

                gradient = np.sign(df_["ndiff"].to_numpy())
                gradkey = 'gradient-%s' % key 
                df_[gradkey]=pd.Series(gradient)
                df_.rename(columns={0:key}, inplace=True)
            
                del df_['ndiff']
                del df_['diff']
                del df_['regime']

                # print("df_:", df_.columns.values.tolist()) 
                df_.plot(ax=ax, grid=True)
            plt.show()

        return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--plot_vs_time", help="select key(s) to plot (ex. \"Field[;Ucoil1]\")")
    parser.add_argument("--plot_key_vs_key", help="select pair(s) of keys to plot (ex. \"Field-Icoil1")
    parser.add_argument("--output_time", help="output key(s) for time")
    parser.add_argument("--output_timerange", help="set time range to extract (start;end)")
    parser.add_argument("--output_key", help="output key(s) for time")
    parser.add_argument("--extract_pairkeys", help="dump key(s) to file")
    parser.add_argument("--show", help="display graphs (default save in png format)", action='store_true')
    parser.add_argument("--list", help="list key in csv", action='store_true')
    parser.add_argument("--convert", help="convert file to csv", action='store_true')
    parser.add_argument("--stats", help="display stats and find regimes", action='store_true')
    args = parser.parse_args()

    # load df pandas from input_file
    mrun = MagnetRun(filename=args.input_file)
    dkeys = mrun.getKeys()

    if args.list:
        print("Valid keys are:")
        for key in dkeys:
            print(key)
        sys.exit(0)

    # perform operations defined by options
    if args.plot_vs_time:
        my_ax = plt.gca()
        # split into keys
        items = args.plot_vs_time.split(';')
        print("items=", items)
        # loop over key
        for key in items:
            if key in dkeys:
                mrun.getMData().plotData(x='Time', y=key, ax=my_ax)
        if args.show:
            plt.show()
        else:
            imagefile = args.input_file.replace(".txt", "")
            plt.savefig('%s_vs_time.png' % imagefile, dpi=300 )
            plt.close()

    if args.plot_key_vs_key:
        # split pairs in key1, key2
        print("plot_key_vs_key=", args.plot_key_vs_key)
        pairs = args.plot_key_vs_key.split(';')
        for pair in pairs:
            print("pair=", pair)
            my_ax = plt.gca()
            #print("pair=", pair, " type=", type(pair))
            items = pair.split('-')
            if len(items) != 2:
                print("invalid pair of keys: %s" % pair)
                sys.exit(1)
            key1= items[0]
            key2 =items[1]
            if key1 in dkeys and key2 in dkeys:
                mrun.getMData().plotData(x=key1, y=key2, ax=my_ax) # on graph per pair
            else:
                raise Exception("unknown keys: %s %s" % (key1, key2), " (Check valid keys with --list option)")
            if args.show:
                plt.show()
            else:
                imagefile = args.input_file.replace(".txt", "")
                plt.savefig('%s_%s_vs_%s.png' % (imagefile, key1, key2), dpi=300 )
                plt.close()

    if args.output_time:
        if mrun.getType():
            print("output_time: feature not implemented for tdms format")
            sys.exit(0)

        times = args.output_time.split(";")
        print ("Select data at %s " % (times) )
        df = mrun.getData()
        if args.output_key:
            keys = args.output_key.split(";")
            print(df[df['Time'].isin(times)][keys])
        else:
            print(df[df['Time'].isin(times)])

    if args.output_timerange:
        if mrun.getType():
            print("output_time: feature not implemented for tdms format")
            sys.exit(0)

        timerange = args.output_timerange.split(";")
        selected_df = mrun.extractTimeData(timerange)

        file_name = args.input_file.replace(".txt", "")
        file_name = file_name + "_from" + str(timerange[0].replace(":", "-"))
        file_name = file_name + "_to" + str(timerange[1].replace(":", "-")) + ".csv"

        df = mrun.getData()
        selected_df=df[df['Time'].between(timerange[0], timerange[1], inclusive=True)]
        if args.output_key:
            keys = args.output_key.split(";")
            keys.insert(0, 'Time')
            selected_df[keys].to_csv(file_name, sep=str('\t'), index=False, header=True)
        else:
            selected_df.to_csv(file_name, sep=str('\t'), index=False, header=True)

    if args.extract_pairkeys:
        if mrun.getType():
            print("output_time: feature not implemented for tdms format")
            sys.exit(0)

        pairs = args.extract_pairkeys.split(';')
        for pair in pairs:
            items = pair.split('-')
            if len(items) != 2:
                print("invalid pair of keys: %s" % pair)
                sys.exit(1)
            key1= items[0]
            key2 =items[1]
            newdf = mrun.extractData(key1, key2)

            # Remove line with I=0
            newdf = newdf[newdf[key1] != 0]
            newdf = newdf[newdf[key2] != 0]

            file_name=str(pair)+".csv"
            newdf.to_csv(file_name, sep=str('\t'), index=False, header=False)

    if args.stats:
        mrun.stats()