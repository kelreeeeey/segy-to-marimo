import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/slides.slides.json",
    sql_output="polars",
)


@app.cell(hide_code=True)
def _(mo):
    section1_header = mo.md(r"""
    # Hello! 😊
    """)
    section1_header.center()
    return


@app.cell
def _(mo):
    subsection1_header = mo.md(r"""
    ## A little about me, to give some context
    """)
    subsection1_header.center()
    return


@app.cell(hide_code=True)
def _(mo):
    intro = mo.md(r"""

    - My name is Rey, I'm from Indoneisa

    - I study Geophysics for my bachelor degree in Universitas Gadjah Mada, and have been intersted in computational geoscience and application of Deep Learning in subsurface data analysis.

    - I'm currently working as research assistant, co-lead a small team of geoscienctists along side with software engineers, building & integrating deep learning for subsurface geoscience application.

    - I'm also preparing for a master degree in Geology.

    """)

    intro
    return


@app.cell(hide_code=True)
def _(mo):
    section2_header = mo.md(r"""
    ## My experience so far with marimo in a few words
    """)
    section2_header.center()
    return


@app.cell
def _(mo):
    akhsay_post = mo.image(src="public/Akhsay_tweet.png", width="50%")
    _first_encounter = mo.vstack(
        [
            mo.hstack(
                [
                    mo.md(rf"""
    - [ ] First encounter: Akhsay's Tweet about the _"The reproducibility of Jupyter Notebook"_
        
    - [ ] I've been using marimo since version `0.8/0.9`ish

    - [ ] I used marimo to do analysis when i'm working on my undergrad thesis

    - [ ] And for past few months, i've been slowly integrating marimo to my job, as well as intrducing it to my peers.

    - [ ] in love with marimo's YouTube videos. Shout out for Vincent!!! 👏👏

    """),
                    mo.vstack(
                        [
                            akhsay_post,
                            mo.md(
                                '<a href="https://x.com/akshaykagrawal/status/1871352603516583959?s=20">link to post</a>'
                            ),
                        ],
                        align="center",
                    ),
                ],
                align="center",
                widths=[0.6, 0.65],
            ),
        ]
    )

    _first_encounter
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    I really like how marimo help me working with\n
    __multi-dimensional__ data more easily through its widgets\n
    and its nature of interactivity.

    """).callout().center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3D Geophysical Data, Seismic to be specific.
    """).center()
    return


@app.cell(hide_code=True)
def _(mo):
    _segy_data_tldr = mo.md("""
    - Developed by Society of Exploration Geophysicists (SEG), 
    - meant to storing geophysical data,
    - SEG-Y Files are stored in a hierarchical byte-stream format that combines both textual and binary data segments.
        """)

    _segy_images = mo.hstack(
        [
            mo.image("public/seismic_data_acquisition.png", width="50%", rounded=True),
            mo.image("public/seismic_trace_illustration.png", width="50%", rounded=True),
        ]
    )
    mo.vstack(
        [
            mo.md(rf"""## `SEGY-Y` Data Format""").center(),
            _segy_data_tldr,
            _segy_images,
        ]
    )
    return


@app.cell
def _(mo):
    _segy_data_tldr = mo.md("""source: SEG (2023)""")
    mo.vstack(
        [
            mo.md(rf"""## `SEGY-Y` Data Format Format""").center(),
            mo.image("public/segy_byte_stream_rev2.1.png", width="80%", rounded=True),
            _segy_data_tldr,
            mo.image("public/seismic_trace_illustration.png", width="55%", rounded=True),
        ],
        align="center"
    )
    return


@app.cell
def _(mo):
    _segy_data_tldr = mo.md("""

    <a style='color:red'>___This kind of data are often___</a>

    1. **Big, (resource related)**

    2. **Stored in byte-stream, not very accessible in Python, (decoding problem)**

    """)
    mo.vstack(
        [
            mo.md(rf"""## `SEGY-Y` Data Format Format""").center(),
            _segy_data_tldr,
        ],
        align="center"
    )
    return


@app.cell
def _(mo):
    _segy_data_tldr = mo.md("""

    **For multi-dimensional data, we have to know some specific
    information about the data to be able to transform the
    1D byte stream to N-D array**

    """).callout(kind="warn")

    _img = mo.image("public/segy_exmple_textual_header.png", width="55%", rounded=True)
    mo.vstack(
        [
            mo.md(rf"""## `SEGY-Y` Data Format Format""").center(),
            mo.hstack([_segy_data_tldr, _img], align="center", justify="end"),
        ],
        align="center"
    )
    return


@app.cell
def _(mo):
    mo.md("""

    **Some small demo :)**

    """).callout(kind="warn").center()
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("""

    ## [**`threey`**](https://github.com/kelreeeeey/threey)

    > a new widget :D, as a gift for marimo & this community!

    """),
            mo.image("public/marimo - threey.png", width="80%").center(),
            mo.md("""

    - [demo on molab](https://molab.marimo.io/notebooks/nb_ctHpuj4ycr8WBrmPQqUyqY),   
    - [demo on my github page](https://kelreeeeey.github.io/marimo-gh-pages/seismic_data_preprocessing_apps/seismic_3d_viewer.html)

    """),
        ], align="center"
    ).callout(kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # That's all thankyou! 😊
    """).center()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
