import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns", sql_output="polars")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Utility, more like a small wrapper

    source: https://github.com/kelreeeeey/segy-to-marimo

    it only needs:
    - `numpy`
    - `polars`
    - [`segfast`](https://github.com/analysiscenter/segfast)
    - matplotlib & threey (optional)

    Also Marimo Snippet Snippets:

    Two snippets I use in this demo: https://github.com/kelreeeeey/segy-to-marimo/tree/main/snippets
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    the_wrapper = mo.watch.file("./thewrapper.py")

    mo.ui.code_editor(the_wrapper.read_text(), language="python", theme="dark")
    return


@app.cell(column=1)
def _(mo):
    mo.md(r"""
    # Demo 1 - Working with SEG-Y Data in Python
    """).center()
    return


@app.cell
def _(Path, mo):
    SEGYDATA_PATH = list(Path("./example-data").rglob("*.sgy")) + list(Path("./example-data").rglob("*.segy"))
    SEGYDATA_PATH = {x.name:x for x in SEGYDATA_PATH}
    SEGY_FILE_SELECTIONS = mo.ui.dropdown(SEGYDATA_PATH, label="Select SEGY File", searchable=True)
    return (SEGY_FILE_SELECTIONS,)


@app.cell
def _(SEGY_FILE_SELECTIONS):
    from thewrapper import inspect_trace_header, get_textual_header

    segy_data = SEGY_FILE_SELECTIONS.value # change this to your data source
    return get_textual_header, inspect_trace_header, segy_data


@app.cell
def _(get_textual_header, inspect_trace_header, mo, segy_data):
    mo.stop(not segy_data)
    text_header = get_textual_header(segy_data, header_encoding="ISO-8859-1")
    non_zero_headers = inspect_trace_header(segyfile=segy_data, override_cache=False)
    non_zero_headers_options = non_zero_headers.columns
    UI_INLINE_3D_MAP_TO    = mo.ui.dropdown(options=non_zero_headers_options, label="Map `INLINE_3D` to")
    UI_CROSSLINE_3D_MAP_TO = mo.ui.dropdown(options=non_zero_headers_options, label="Map `CROSSLINE_3D` to")
    UI_COORDINATE_X_MAP_TO = mo.ui.dropdown(options=non_zero_headers_options, label="Map `CDP_X` to")
    UI_COORDINATE_Y_MAP_TO = mo.ui.dropdown(options=non_zero_headers_options, label="Map `CDP_Y` to")
    MIN_IL = mo.ui.number(value=0, step=1, stop=1000000, label="Minimum INLINE")
    MIN_XL = mo.ui.number(value=0, step=1, stop=1000000, label="Minimum CROSSLINE")
    return (
        MIN_IL,
        MIN_XL,
        UI_COORDINATE_X_MAP_TO,
        UI_COORDINATE_Y_MAP_TO,
        UI_CROSSLINE_3D_MAP_TO,
        UI_INLINE_3D_MAP_TO,
        non_zero_headers,
        text_header,
    )


@app.cell
def _(SEGY_FILE_SELECTIONS):
    SEGY_FILE_SELECTIONS.center()
    return


@app.cell(hide_code=True)
def _(mo, non_zero_headers, segy_data, text_header):
    mo.stop(not segy_data)
    mo.vstack(
        [
            mo.md("""## Header and Trace Geometry Inspections. <b style="color:red">Please read the Textual header first!</b>
            ---

            **You only have to look for column with unique values that are reasonable for cube's lateral geometry**
            """),
            mo.vstack([
                mo.ui.text_area(text_header, rows=20),
                mo.ui.table(non_zero_headers, page_size=10) if not isinstance(non_zero_headers, type(None)) else None
            ]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    MIN_IL,
    MIN_XL,
    UI_COORDINATE_X_MAP_TO,
    UI_COORDINATE_Y_MAP_TO,
    UI_CROSSLINE_3D_MAP_TO,
    UI_INLINE_3D_MAP_TO,
    mo,
    non_zero_headers,
):
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([UI_INLINE_3D_MAP_TO, MIN_IL])
                    if not UI_INLINE_3D_MAP_TO.value or not MIN_IL.value
                    else mo.vstack(
                        [
                            UI_INLINE_3D_MAP_TO,
                            MIN_IL,
                            mo.md(f"---\n\n### {UI_INLINE_3D_MAP_TO.value}"),
                            non_zero_headers[UI_INLINE_3D_MAP_TO.value].describe(),
                        ]
                    ),
                    mo.vstack([UI_CROSSLINE_3D_MAP_TO, MIN_XL])
                    if not UI_CROSSLINE_3D_MAP_TO.value or not MIN_XL.value
                    else mo.vstack(
                        [
                            UI_CROSSLINE_3D_MAP_TO,
                            MIN_XL,
                            mo.md(f"---\n\n### {UI_CROSSLINE_3D_MAP_TO.value}"),
                            non_zero_headers[UI_CROSSLINE_3D_MAP_TO.value].describe(),
                        ]
                    ),
                ],
                widths=[0.5, 0.5],
            ),
            mo.hstack(
                [
                    UI_COORDINATE_X_MAP_TO
                    if not UI_COORDINATE_X_MAP_TO.value
                    else mo.vstack(
                        [
                            UI_COORDINATE_X_MAP_TO,
                            mo.md(f"---\n\n### {UI_COORDINATE_X_MAP_TO.value}"),
                            non_zero_headers[UI_COORDINATE_X_MAP_TO.value].describe(),
                        ]
                    ),
                    UI_COORDINATE_Y_MAP_TO
                    if not UI_COORDINATE_Y_MAP_TO.value
                    else mo.vstack(
                        [
                            UI_COORDINATE_Y_MAP_TO,
                            mo.md(f"---\n\n### {UI_COORDINATE_Y_MAP_TO.value}"),
                            non_zero_headers[UI_COORDINATE_Y_MAP_TO.value].describe(),
                        ]
                    ),
                ],
                widths=[0.5, 0.5],
            ),
        ]
    ).callout()
    return


@app.cell
def _(
    UI_COORDINATE_X_MAP_TO,
    UI_COORDINATE_Y_MAP_TO,
    UI_CROSSLINE_3D_MAP_TO,
    UI_INLINE_3D_MAP_TO,
    mo,
):
    mo.stop(not (None != UI_COORDINATE_X_MAP_TO.value and None != UI_COORDINATE_Y_MAP_TO.value))
    mo.stop(not (None != UI_INLINE_3D_MAP_TO.value and None != UI_CROSSLINE_3D_MAP_TO.value))
    geometry_identifier = {
        "INLINE_3D": UI_INLINE_3D_MAP_TO.value,
        "CROSSLINE_3D": UI_CROSSLINE_3D_MAP_TO.value,
    }
    coordinates_mapper = {
        "CDP_X": UI_COORDINATE_X_MAP_TO.value,
        "CDP_Y": UI_COORDINATE_Y_MAP_TO.value,
    }
    return coordinates_mapper, geometry_identifier


@app.cell
def _(MIN_IL, MIN_XL, coordinates_mapper, geometry_identifier, segy_data):
    from thewrapper import get_3d_cube_geometries, get_3d_cube
    import matplotlib.pyplot as plt

    # EXAMPLE trace header mapping to be passed into `get_3d_cube_geometries`
    # geometry_identifier = {
    #     "INLINE_3D": "INLINE_3D",
    #     "CROSSLINE_3D": "CROSSLINE_3D"
    # }
    # coordinates_mapper = {
    #     "CDP_X": "CDP_X",
    #     "CDP_Y": "CDP_Y"
    # }
    # min_il_xl -> tuple of 2 ints, indicating minimum values for both
    # inline and crossline, usually can be achieved from textual header or
    # trace headers.

    if segy_data:
        cube_geometries = get_3d_cube_geometries(
            segy_data, # change this tho your data source
            geometry_identifier=geometry_identifier,
            coordinates_mapper=coordinates_mapper,
            min_il_xl=(MIN_IL.value, MIN_XL.value)
        )
        trace_header, cube_shape, local_coords, global_coords, trace_sequence = cube_geometries
        print(type(cube_geometries), cube_geometries._fields)
    return (
        cube_shape,
        get_3d_cube,
        global_coords,
        local_coords,
        plt,
        trace_header,
        trace_sequence,
    )


@app.cell
def _(mo, trace_header):
    mo.vstack([mo.md("Fetched Trace Header"), trace_header])
    return


@app.cell(hide_code=True)
def _(global_coords, local_coords, mo, np, plt):
    _f, _ax = plt.subplots(1, 2, figsize=(10, 10), layout="tight")
    _x, _y = np.meshgrid(local_coords.crosslines, local_coords.inlines)
    _f.suptitle("Crossline (left) and Inline (right)")
    _ax[0].scatter(global_coords.xcoords, global_coords.ycoords, c=_x.flatten(), s=2, cmap="jet")
    _ax[0].grid()
    _ax[0].set_xlabel("EASTING"); _ax[0].set_ylabel("NORTHING")
    _ax[1].scatter(global_coords.xcoords, global_coords.ycoords, c=_y.flatten(), s=2, cmap="jet")
    _ax[1].grid()
    _ax[1].set_xlabel("EASTING"); _ax[1].set_ylabel("NORTHING")
    mo.as_html(plt.gcf()).center()
    return


@app.cell(hide_code=True)
def _(global_coords, mo, plt, trace_sequence):
    plt.figure(figsize=(5, 10), layout="tight")
    plt.title("Seismic Traces Position")
    plt.scatter(global_coords.xcoords, global_coords.ycoords, c=trace_sequence.flatten(), s=2, cmap="jet")
    plt.grid()
    plt.xlabel("EASTING"); plt.ylabel("NORTHING")
    traces_plot = mo.as_html(plt.gcf()).center()
    return (traces_plot,)


@app.cell
def _(cube_shape):
    cube_shape
    return


@app.cell
def _(segy_data):
    segy_data
    return


@app.cell
def _(cube_shape, get_3d_cube, segy_data):
    seismic_3d, seismic_metadata = get_3d_cube(segy_data, *cube_shape)
    # print(type(seismic_3d))
    # print(type(seismic_metadata), seismic_metadata)
    return (seismic_3d,)


@app.cell
def _(seismic_3d):
    seismic_3d.shape
    return


@app.cell
def _(global_coords, mo, plt, seismic_3d):
    plt.figure(figsize=(5, 10), layout="tight")
    plt.title("Seismic Depth Slice")
    plt.scatter(global_coords.xcoords, global_coords.ycoords, c=seismic_3d[:, :, 100].flatten(), s=2, cmap="seismic")
    # plt.imshow(seismic_3d[:, :, 100].T, cmap="seismic")
    plt.grid()
    # plt.xlabel("EASTING"); plt.ylabel("NORTHING")
    time_slice_plot = mo.as_html(plt.gcf()).center()
    return (time_slice_plot,)


@app.cell
def _(mo, time_slice_plot, traces_plot):
    mo.hstack([traces_plot, time_slice_plot])
    return


@app.cell
def _(mo, plt, seismic_3d):
    _f, _ax = plt.subplots(1, 2, figsize=(10, 7), layout="tight", sharey=True)
    _ax[0].set_title("Crossline Section")
    _ax[1].set_title("Inline Section")
    _ax[0].imshow(seismic_3d[:, 50, :].T, cmap="seismic", aspect="equal")
    _ax[0].grid()
    _ax[0].set_xlabel("INLINE INDEX"); _ax[0].set_ylabel("Two Way Travel Time (TWT) Sample")
    _ax[1].imshow(seismic_3d[50, :, :].T, cmap="seismic", aspect="equal")
    _ax[1].grid()
    _ax[1].set_xlabel("CROSSLINE INDEX")
    mo.as_html(plt.gcf()).center()
    return


@app.cell
def _():
    import numpy as np
    import segfast as sg
    import polars as pl
    from pathlib import Path
    return Path, np


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(column=2)
def _(mo):
    mo.md(r"""
    # Demo 2 - threey
    """).center()
    return


@app.cell
def _():
    from threey import Seismic3DViewer
    return (Seismic3DViewer,)


@app.cell
def _(Seismic3DViewer, mo, np, segy_data, seismic_3d):
    mo.stop(not segy_data)
    mo.stop(not isinstance(seismic_3d, np.ndarray))

    vmin, vmax = seismic_3d.data.min(), seismic_3d.data.max()

    # the widget takes memoryview of both the seismic data & the label
    sample_cube = memoryview(seismic_3d.data.transpose((2, 0, 1))[:350, :, :])

    labels = {}
    kwargs_labels = {} # store the colormap and alpha for the label here!

    _dimensions = dict(
        inline=sample_cube.shape[1],
        crossline=sample_cube.shape[2],
        depth=sample_cube.shape[0]
    )

    area = mo.ui.anywidget(
        Seismic3DViewer(
            data_source = sample_cube,
            cmap_data = "seismic", # default to "seismic"
            dark_mode=False if mo.app_meta().theme != "dark" else True,
            labels=labels,
            kwargs_labels=kwargs_labels,
            show_label= False,
            vmin = vmin,
            vmax = vmax,
            is_2d_view = False, # default to True
            dimensions=_dimensions,
            height=500
        )
    )
    area
    return


if __name__ == "__main__":
    app.run()
