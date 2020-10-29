"""MagnetData"""

import os
import sys
import datetime
import matplotlib.pyplot as plt
from nptdms import TdmsFile
import pandas as pd
import numpy as np
import matplotlib
# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available
import file_utils

Supported_Fformat=dict()
Supported_Fformat[".txt"] = file_utils.FileType("txt", sep=r'\s+')
Supported_Fformat[".csv"] = file_utils.FileType("csv", sep=str(","))

# Ensight:
Supported_Fformat[".csv-ensight"] = file_utils.FileType("csv", skiprows=2)

Supported_Fformat[".tdms"] = 1

class MagnetData:
    """
    Magnet Data

    FileName: name of the input file
    Data: pandas dataframe or tdms file
    Keys:
    Groups:
    Type: 0 for Pandas data, 1 for Tdms

    Use post input parameter to force Ensight format
    """

    def __init__(self, filename="tutu", post=""):
        self.FileName = filename
        self.Groups = dict()
        self.Keys = []
        self.Type = 0 # 0 for Pandas, 1 for Tdms

        with open(filename, 'r') as f:
            f_extension=os.path.splitext(filename)[-1]
            if post == "ensight":
                f_extension += "-ensight"
            print("f_extension: % s" % f_extension)
            if f_extension in Supported_Fformat:
                if f_extension == ".tdms":
                    self.Type = 1
                    self.Data = TdmsFile.open(filename, 'r')
                    for group in self.Data.groups():
                        for channel in group.channels():
                            self.Keys.append(channel.name)
                            self.Groups[channel.name] = group.name

                else:
                    # print("Loading pandas")
                    self.Type = 0

                    self.Data = pd.read_csv(f,
                                   sep=Supported_Fformat[f_extension].getSep(),
                                   engine=Supported_Fformat[f_extension].getEngine(),
                                   skiprows=Supported_Fformat[f_extension].getSkipRows())
                    # print("self.Data=", self.Data)
                    self.Keys = self.Data.columns.values.tolist()
                    # print("self.Keys=", self.Keys)

    def __repr__(self):
        return "%s(Type=%r, Groups=%r, Keys=%r, Data=%r)" % \
            (self.__class__.__name__,
             self.Type,
             self.Groups,
             self.Keys,
             self.Data)

    def getType(self):
        """returns Data Type"""
        return self.Type

    def getData(self):
        """retunr Data"""
        return self.Data

    def getKeys(self):
        """return list of Data keys"""
        return self.Keys

    def cleanupData(self):
        """removes empty columns from Data"""
        if self.Type == 0 :
            self.Data = self.Data.loc[:, (self.Data != 0.0).any(axis=0)]
            self.Keys = self.Data.columns.values.tolist()
        return 0

    def removeData(self, key):
        """remove a column to Data"""
        if self.Type == 0 :
            if key in self.Keys:
                del self.Data[key]
        else:
            raise Exception("cannot remove %s: no such key" % key)
        return 0

    def addData(self, key, formula):
        """
        add a new column to Data from  a formula
        
        see: 
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.eval.html
        https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#enhancingperf-eval
        """

        print("addData: %s = %s" % (key, formula) )
        if self.Type == 0 :
            if key in self.Keys:
                print("Key %s already exists in DataFrame")
            else:
                # check formula using pyparsing
                
                self.Data.eval(formula, inplace=True)
        return 0

    def addTime(self):
        """add a Time column to Data"""

        if self.Type == 0 :
            if "Date" in self.Keys and "Time" in self.Keys:
                tformat="%Y.%m.%d %H:%M:%S"
                start_date=self.Data["Date"].iloc[0]
                start_time=self.Data["Time"].iloc[0]
                end_time=self.Data["Time"].iloc[-1]
                print ("start_time=", start_time, "start_date=", start_date)

                t0 = datetime.datetime.strptime(self.Data['Date'].iloc[0]+" "+self.Data['Time'].iloc[0], tformat)
                self.Data["t"] = self.Data.apply(lambda row: (datetime.datetime.strptime(row.Date+" "+row.Time, tformat)-t0).total_seconds(), axis=1)
            else:
                raise Exception("cannot add t[s] columnn: no Date or Time column")
        return 0

    def extractData(self, key1, key2):
        """extract columns key1 and key2 to Data"""

        newdf = None
        if self.Type == 0 :
            if key1 in self.Keys and key2 in self.Keys:
                newdf = pd.concat([self.Data[key1], self.Data[key2]], axis=1)
            else:
                raise Exception("cannot extract %s and %s columnns" % (key1, key2))
        return newdf

    def extractTimeData(self, timerange):
        """extract column to Data"""

        selected_df = None
        if self.Type == 0 :
            timerange = args.output_timerange.split(";")
            print ("Select data from %s to %s" % (timerange[0],timerange[1]) )

            selected_df=self.Data[self.Data['Time'].between(timerange[0], timerange[1], inclusive=True)]
        return selected_df

    def saveData(self, keys, filename):
        """save Data to csv format"""
        if self.Type == 0 :
            self.Data[keys].to_csv(filename, sep=str('\t'), index=False, header=True)
            return 0

    def plotData(self, x, y, ax):
        """plot x vs y"""
        
        if not x in self.Keys:
            if self.Type == 0 :
                raise Exception("cannot plot x=%s columnns" % x)
            else:
                if x != "Time" :
                    raise Exception("cannot plot x=%s columnns" % x)

        if y in self.Keys:
            if self.Type == 0 :
                self.Data.plot(x=x, y=y, ax=ax, grid=True)
            else:
                group = self.Data[self.Groups[y]]
                channel = group[y]
                samples = channel.properties['wf_samples']

                if x == 'Time':
                    increment = channel.properties['wf_increment']
                    time_steps = np.array([i*increment for i in range(0,samples)])

                    plt.plot(time_steps, self.Data[self.Groups[y]][y], label=y)
                    plt.ylabel(" [" + channel.properties['unit_string'] + "]")
                    plt.xlabel("t [s]")
                else:
                    group = self.Data[self.Groups[x]]
                    xchannel = group[x]

                    plt.plot(self.Data[self.Groups[x]][x], self.Data[self.Groups[y]][y], label=y)
                    plt.ylabel(" [" + channel.properties['unit_string'] + "]")
                    plt.xlabel(xchannel.name + " [" + xchannel.properties['unit_string'] + "]")

                plt.grid(b=True)
                my_ax.legend()
        else:
            raise Exception("cannot plot y=%s columnns" % y)

    def stats(self):
        """compute stats fro the actual run"""
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
                mrun.plotData(x='Time', y=key, ax=my_ax)
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
                mrun.plotData(x=key1, y=key2, ax=my_ax) # on graph per pair
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
