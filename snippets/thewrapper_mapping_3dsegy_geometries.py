import marimo

app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # segy-to-marimo: Mapping Cube's Geometry and Coordinates to Trace Headers
        """
    )
    return

#magic_start_parse
@app.cell
def _():
    import marimo as mo
    from thewrapper import inspect_trace_header, get_textual_header

    segy_data = None # change this to your data source
    return (segy_data, inspect_trace_header, get_textual_header)


@app.cell
def _(mo, segy_data, inspect_trace_header, get_textual_header):
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
        UI_INLINE_3D_MAP_TO,
        UI_CROSSLINE_3D_MAP_TO,
        UI_COORDINATE_X_MAP_TO,
        UI_COORDINATE_Y_MAP_TO,
        MIN_IL,
        MIN_XL,
    )


@app.cell(hide_code=True)
def _(
    segy_data,
    non_zero_headers,
    text_header,
):
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
    mo,
    UI_INLINE_3D_MAP_TO,
    UI_CROSSLINE_3D_MAP_TO,
    UI_COORDINATE_X_MAP_TO,
    UI_COORDINATE_Y_MAP_TO,
    MIN_IL,
    MIN_XL,
    non_zero_headers
):
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
    )

@app.cell(hide_code=True)
def _(
    mo,
    UI_INLINE_3D_MAP_TO,
    UI_CROSSLINE_3D_MAP_TO,
    UI_COORDINATE_X_MAP_TO,
    UI_COORDINATE_Y_MAP_TO,
    MIN_IL,
    MIN_XL,
    non_zero_headers
):
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
    )
    return

@app.cell
def _(mo, ):
    mo.stop(
        not (
            None != UI_COORDINATE_X_MAP_TO.value and None != UI_COORDINATE_Y_MAP_TO.value
        )
    )
    mo.stop(not (None != UI_INLINE_3D_MAP_TO.value and None != UI_CROSSLINE_3D_MAP_TO.value))
    geometry_identifier = {
        "INLINE_3D": UI_INLINE_3D_MAP_TO.value,
        "CROSSLINE_3D": UI_CROSSLINE_3D_MAP_TO.value,
    }
    coordinates_mapper = {
        "CDP_X": UI_COORDINATE_X_MAP_TO.value,
        "CDP_Y": UI_COORDINATE_Y_MAP_TO.value,
    }
    return (geometry_identifier, coordinates_mapper)

#magic_end_parse


if __name__ == "__main__":
    app.run()


