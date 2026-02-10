import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from irsa.io import load_spectra
    return (load_spectra,)


@app.cell
def _(load_spectra):
    dd = load_spectra("/home/intellimath/repos/spectra/13", {})
    return (dd,)


@app.cell
def _(dd, mo):
    keys = mo.ui.dropdown(
        options=dd.keys(), label="choose one")
    keys

    return (keys,)


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    return mo, plt


@app.cell
def _(dd, keys):
    sp_series = dd[keys.value]
    return (sp_series,)


@app.cell
def _(mo, sp_series):
    i_serie = mo.ui.slider(start=0, stop=len(sp_series.x)-1,show_value=True)
    i_serie    

    return (i_serie,)


@app.cell
def _(i_serie, plt, sp_series):
    x = sp_series.x[i_serie.value]
    ys = sp_series.y[i_serie.value]
    for y in ys:
        plt.plot(x, y)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
