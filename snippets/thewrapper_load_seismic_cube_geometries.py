import marimo

app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # segy-to-marimo: Load seismic cube 3D geometry
        """
    )
    return

#magic_start_parse
@app.cell
def _():
    path_to_segy_data = None
    return path_to_segy_data

@app.cell
def _(path_to_segy_data):
    import marimo as mo
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

    if path_to_segy_data:
        cube_geometries = get_3d_cube_geometries(
            path_to_segy_data, # change this tho your data source
            geometry_identifier={},
            coordinates_mapper={},
            min_il_xl=(0, 0)
        )
        trace_header, cube_shape, local_coords, global_coords, trace_sequence = cube_geometries
        print(type(cube_geometries), cube_geometries._fields)
    return (mo, cube_geometries, get_3d_cube, plt)


@app.cell
def _(trace_header):
    trace_header
    return

@app.cell
def _(plt, global_coords, trace_sequence):
    plt.figure(figsize=(10, 5), layout="tight")
    plt.title("Seismic Traces Position")
    plt.scatter(global_coords.xcoords, global_coords.ycoords, c=trace_sequence, s=2, cmap="jet")
    plt.grid()
    plt.xlabel("EASTING"); plt.ylabel("NORTHING")
    plt.show()
    return


@app.cell
def _(cube_shape):
    cube_shape
    return


@app.cell
def _(get_3d_cube, path_to_segy_data, cube_shape):
    seismic_3d, seismic_metadata = get_3d_cube(path_to_segy_data, *cube_shape)
    print(type(seismic_3d))
    print(type(seismic_metadata), seismic_metadata)
    return (seismic_3d, )
#magic_end_parse

@app.cell
def _(plt, global_coords, trace_sequence):
    plt.figure(figsize=(10, 5), layout="tight")
    plt.title("Seismic Time Slice")
    plt.scatter(global_coords.xcoords, global_coords.ycoords, c=trace_sequence, s=2, cmap="jet")
    plt.grid()
    plt.xlabel("EASTING"); plt.ylabel("NORTHING")
    plt.show()
    return



if __name__ == "__main__":
    app.run()
