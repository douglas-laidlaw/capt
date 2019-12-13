import matplotlib
matplotlib.rc('font', **{'size':20, 'family': 'serif', 'serif':['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['image.cmap'] = 'magma'
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['image.interpolation'] = 'nearest'
matplotlib.rcParams['figure.autolayout'] = True
matplotlib.rc('legend', **{'fontsize':14})
