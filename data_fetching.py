import marimo

__generated_with = "0.18.3"
app = marimo.App(width="full", sql_output="polars")


@app.cell
def _():
    import requests as req
    import os
    import json
    import marimo as mo
    return json, mo, os, req


@app.cell
def _(os):
    datadir = os.path.join(os.getcwd(), "example-data")
    if not os.path.exists(datadir):
        os.makedirs(datadir, exist_ok=True)
    return (datadir,)


@app.cell
def _(mo):
    fetch_button = mo.ui.run_button(label="Fetch Selected Data", kind="warn")
    return (fetch_button,)


@app.cell
def _(mo):
    mo.md(r"""
    ## _fetching sample data_
    """)
    return


@app.cell
def _(json, mo, datadir):
    with (
        open(os.path.join(datadir, "data_segysak_urls.json"), "r") as _f,
        open(os.path.join(datadir, "open_data_new_zealand.json"), "r") as _ff,
    ):
        urls  = json.load(_f)
        urls |= json.load(_ff)

    (select_data := mo.ui.table([
        {"name":key, "link": value} for key, value in urls.items()
    ]))
    return (select_data,)


@app.cell
def _(fetch_button, mo, select_data):
    mo.stop(not select_data.value, mo.md("Select data to fetch!"))
    fetch_button.center()
    return


@app.cell
def _(datadir, fetch_button, mo, os, req, select_data):
    mo.stop(not select_data.value, mo.md("Select data to fetch!"))
    _mode = mo.app_meta().mode
    if fetch_button.value:
        for _file_name, _url in map(lambda x: (x['name'], x['link']), select_data.value):
            _data_path = os.path.join(datadir, _file_name)
            if not os.path.exists(_data_path):
                if _mode == "edit" or "run" == _mode: 
                    mo.output.append(mo.md(f"- Fetching `{_file_name}` and writing into :\n\t`{_data_path}`"))
                else:
                    print("Fetching", _file_name, "and writing into:\n\t", _data_path)
                _requests = req.get(_url)
                with open(os.path.join(datadir, _file_name), "wb") as _f:
                    _f.write(_requests.content)
            else:
                if _mode == "edit" or "run" == _mode: 
                    mo.output.append(mo.md(f"- `{_file_name}` already exists at:\n\t`{_data_path}`"))
                else:
                    print(_file_name, "already exists at:\n\t", _data_path)
    return


if __name__ == "__main__":
    app.run()
