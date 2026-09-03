import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, plt


@app.cell
def _():
    import mlgrad.funcs as funcs
    import mlgrad.smooth as smooth
    import mlgrad.inventory as inventory

    from irsa.io import load_spectra
    from sklearn.decomposition import PCA, FastICA

    return PCA, funcs, inventory, load_spectra


@app.cell
def _(load_spectra):
    dd = load_spectra("/home/intellimath/repos/spectra/20", 
                      {
                        "вид_бактерий":["SA", "SS", "SH"], 
                        "номер_цикла":["20"], 
                        "лизис":"термический",
                        # "номер_эксперимента_в_цикле":["2", "4"],
                        "отсечки_по_молекулярной_массе": ["super"],                      
                      }, verbose=False)
    return (dd,)


@app.cell
def _(dd):
    dd.align_bottom()
    dd.crop(300)
    dd.scale_by_mean()
    return


@app.cell
def _(dd):
    dd.drop_zeros_spectrum()
    dd.drop_notfinite()
    dd.drop_empty()
    dd.isfinite()
    return


@app.cell
def _(dd):
    dd.flatten()
    return


@app.cell
def _(dd, mo):
    keys = mo.ui.dropdown(
        options=dd.keys(), label="choose one")
    keys
    return (keys,)


@app.cell
def _(dd, keys):
    sp_series = dd[keys.value]
    return (sp_series,)


@app.cell
def _(mo, sp_series):
    i_serie = mo.ui.slider(start=0, stop=len(sp_series.x)-1,show_value=True)
    i_serie    
    return


@app.cell
def _(plt, sp_series):
    x = sp_series.x
    ys = sp_series.y
    plt.figure(figsize=(12,4))
    for y in ys:
        plt.plot(x, y, color='gray', alpha=0.2)
    plt.plot(x, ys.mean(axis=0), color='k', lw=1.0)
    plt.grid(ls=':', lw=0.75)
    plt.minorticks_on()
    plt.show()
    return (ys,)


@app.cell
def _(dd, funcs, inventory):
    dd.subtract_baselines(tau2=1.0e4,     
                          func=funcs.RStep(0.0, 0.001),
                          # func2=funcs.RStep(0.001, 0.001),
                          func2_e=inventory.relative_abs_max,
                          verbose=False,
    )
    return


@app.cell
def _(PCA, plt, ys):
    pca = PCA(10)
    us = pca.fit_transform(ys)
    ratio = (us[:,0].max() - us[:,0].min()) / (us[:,1].max() - us[:,1].min())
    plt.figure(figsize=(10, 10/ratio))
    plt.scatter(us[:,0], us[:,1])
    plt.minorticks_on()
    plt.grid(ls=':')
    plt.show()

    plt.plot(pca.explained_variance_ratio_, marker='o')
    plt.minorticks_on()
    plt.grid(ls=':')
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
