#
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
import os.path


parser = argparse.ArgumentParser(
    description="view residual history for a steady computation"
)

parser.add_argument("-n", "--name", default="", type=str)
parser.add_argument("-i", "--index", default=0, type=int)
parser.add_argument("-p", "--prefix", default="data/out/", type=str)
parser.add_argument("-s", "--see", default=2, type=int)
parser.add_argument("-o", "--output", default="cur.png", type=str)
parser.add_argument("-t", "--title", default="", type=str)
parser.add_argument("--normalize", action="store_true", default=False)
parser.add_argument("--nlogy", action="store_true", default=False)
parser.add_argument("--force", action="store_true", default=False)
parser.add_argument("--ylim", default="", type=str)
parser.add_argument("--xlim", default="", type=str)


args = parser.parse_args()
print(args)

mode = ""
if len(str(args.name)):
    mode = "name"
else:
    mode = "index"

names = os.listdir(args.prefix)
names = filter(lambda x: re.match(r".*\.log", x), names)

fileDirs = {}

for name in names:
    namefull = os.path.join(args.prefix, name)
    stat = os.stat(namefull)
    fileDirs[namefull] = stat.st_mtime  # sort with mtime or ctime
    # print(stat)

fileDirsSorted = sorted(
    fileDirs.items(), key=lambda x: x[1], reverse=True
)  # latest first

fname = ""
filetime = 0

if mode == "name":
    for pair in fileDirsSorted:
        fname = pair[0]
        filetime = pair[1]
        print(fname)
        if re.match(args.name, fname):
            break
    else:
        raise RuntimeError("No such file as: " + args.name)

elif mode == "index":
    assert args.index < fileDirsSorted.__len__()
    fname = fileDirsSorted[args.index][0]
    filetime = fileDirsSorted[args.index][1]
else:
    raise ValueError("no such mode")

isee = args.see

foutpic = os.path.join(args.prefix, args.output)
print("Plotting error number %d at file [%s] to [%s]" % (isee, fname, foutpic))

if os.path.exists(foutpic):
    stat = os.stat(foutpic)
    if stat.st_ctime >= filetime:
        print("Output file seems newer")
        if not args.force:
            print("Exiting")
            exit(0)


fin = open(fname, "r")
lines = fin.readlines()
fin.close()

headline = lines[0]
nData = headline.split().__len__()
nLines = lines.__len__()

dataIn = np.zeros([nLines, nData])

for iLine in range(nLines-1):
    dataIn[iLine, :] = np.array(list(map(lambda x: float(x), lines[iLine].split())))

dataInInner = dataIn[dataIn[:, 1] > 0, :]


fig = plt.figure(1, figsize=np.array([4, 3]) * 2, facecolor="white")

fig.set_frameon(True)
ax = plt.axes()

ax.grid("both")
ax.set_xlabel("n_iterin")
if args.nlogy:
    ax.set_yscale("linear")
else:
    ax.set_yscale("log")
if len(args.title):
    ax.set_title(args.title)
else:
    ax.set_title("V_%d file [%s]" % (isee, fname))
if len(args.ylim):
    lims = eval(args.ylim)
    ax.set_ylim(lims[0], lims[1])

if len(args.xlim):
    lims = eval(args.xlim)
    ax.set_xlim(lims[0], lims[1])

dataPlotY = dataInInner[:, isee]
if args.normalize:
    dataPlotY /= np.max(dataInInner[:, isee])
plt.plot(dataInInner[:, 1], dataPlotY)
plt.plot(dataInInner[-1, 1], dataPlotY[-1], "o")
print("Number of Entries %d" % (len(dataPlotY)))

plt.draw()
fig.savefig(foutpic)


#
