import matplotlib

text_width = 5.50107  # in  --> Confirmed with template explanation
dpi = 1200

fs_m1 = 18  # for figure ticks
fs_leg = 18 # for legend
fs = 18  # for regular figure text
fs_p1 = 25  #  figure titles

matplotlib.rc("font", size=fs)  # controls default text sizes
matplotlib.rc("axes", titlesize=fs)  # fontsize of the axes title
matplotlib.rc("axes", labelsize=fs)  # fontsize of the x and y labels
matplotlib.rc("xtick", labelsize=fs_m1)  # fontsize of the tick labels
matplotlib.rc("ytick", labelsize=fs_m1)  # fontsize of the tick labels
matplotlib.rc("legend", fontsize=fs_leg)  # legend fontsize
matplotlib.rc(
    "figure", titlesize=fs_p1, dpi=dpi, autolayout=True
)  # fontsize of the figure
matplotlib.rc("lines", linewidth=3, markersize=3)
matplotlib.rc("savefig", dpi=1200, bbox="tight")
matplotlib.rc("grid", alpha=0.3)
matplotlib.rc("axes", grid=True)

matplotlib.rc("font", **{"family": "serif", "serif": ["Palatino"]})

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Palatino", 
        "Palatino Linotype",  
        "URW Palladio L",      
        "TeX Gyre Pagella",    
        "Book Antiqua",
        "DejaVu Serif"         
    ]
})
#matplotlib.rc("text", usetex=True)