#

def plot_spectra(x, ys, i_sel, color_sel="darkblue", lw=0.5, lw_sel=1.0, marker=None, ms=0):
    for y in ys:
        plt.plot(x, y, linewidth=lw, alpha=0.25, marker=marker, ms=ms)

