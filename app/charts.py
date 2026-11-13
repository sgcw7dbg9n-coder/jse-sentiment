import re

import plotly.graph_objects as go
import streamlit as st

# Categorical slots in fixed order, light and dark steps of the same hues (validated with the dataviz palette script)
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
# Marks that are context rather than a compared entity (prices, headline counts, the baseline model)
NEUTRAL = "#898781"

SCORERS = {
    "vader_score": "VADER",
    "finbert_score": "FinBERT",
    "lm_score": "Loughran-McDonald",
    "gdelt_tone": "GDELT tone",
}
# Colour follows the scorer everywhere, so FinBERT is orange on every page whatever else is on screen
SLOTS = {"vader_score": 0, "finbert_score": 1, "lm_score": 2, "gdelt_tone": 3, "all_sentiment": 4}

FEATURE_SETS = {
    "baseline": "Prices only (baseline)",
    "all_sentiment": "+ all four scorers",
    **{score: f"+ {label}" for score, label in SCORERS.items()},
}
MODELS = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "svm_rbf": "SVM (RBF kernel)"}
# For use mid-sentence, where only the acronym keeps its capitals
MODELS_INLINE = {"logistic_regression": "logistic regression", "random_forest": "random forest",
                 "svm_rbf": "SVM (RBF kernel)"}


def target_label(target: str) -> str:
    horizon = int(re.search(r"_(\d+)d$", target).group(1))
    days = "next session" if horizon == 1 else f"next {horizon} sessions"
    if "excess" in target:
        return f"Beats the Satrix 40 over the {days}"
    return f"Price up over the {days}"


def is_dark() -> bool:
    theme = getattr(getattr(st, "context", None), "theme", None)
    return getattr(theme, "type", None) == "dark"


def colour(key: str) -> str:
    return (DARK if is_dark() else LIGHT)[SLOTS[key]] if key in SLOTS else NEUTRAL


def surface() -> str:
    # Matches backgroundColor in .streamlit/config.toml, so marker rings read as gaps
    return "#1a1a19" if is_dark() else "#fcfcfb"


def diverging_scale() -> list[list]:
    blue, red = colour("vader_score"), (DARK if is_dark() else LIGHT)[7]
    middle = "#383835" if is_dark() else "#f0efec"
    return [[0.0, red], [0.5, middle], [1.0, blue]]


def style(fig: go.Figure, height: int, hovermode: str = "closest", legend: bool = True) -> go.Figure:
    # Room for the title, plus a row for a legend that sits between the title and the plot
    top = (40 if fig.layout.title.text else 16) + (36 if legend else 0)
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=top + 16, b=8),
        title=dict(yref="container", y=1, yanchor="top", pad=dict(t=12)),
        hovermode=hovermode,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        bargap=0.35,
        barcornerradius=4,
    )
    grid, axis = ("#2c2c2a", "#383835") if is_dark() else ("#e1e0d9", "#c3c2b7")
    shared = {"zeroline": False, "gridwidth": 1, "gridcolor": grid, "linewidth": 1, "linecolor": axis, "ticks": "",
              "tickfont.color": NEUTRAL}
    spikes = dict(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="solid",
                  spikecolor=NEUTRAL) if hovermode.startswith("x") else {}
    fig.for_each_xaxis(lambda ax: _defaults(ax, showline=True, **shared, **spikes))
    fig.for_each_yaxis(lambda ax: _defaults(ax, **shared))
    return fig


def _defaults(axis, **props) -> None:
    # Fill in only what the chart left unset, so chart-specific axis settings win whatever the call order
    for name, value in props.items():
        if axis[name] is None:
            axis[name] = value


def legend_on_row(fig: go.Figure, row: int) -> go.Figure:
    # In stacked subplots the legend sits on the row it describes rather than above the whole figure
    top = fig.layout[f"yaxis{row}"].domain[1]
    return fig.update_layout(legend=dict(y=top, yanchor="bottom", x=1, xanchor="right"))


def reference_line(fig: go.Figure, axis: str, value: float, **subplot) -> None:
    # Reference lines are chrome, so they take the axis colour rather than the neutral data colour
    add = fig.add_hline if axis == "y" else fig.add_vline
    add(**{axis: value}, line_width=1, line_color="#383835" if is_dark() else "#c3c2b7", layer="below", **subplot)


def diagonal(fig: go.Figure) -> None:
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, layer="below",
                  line=dict(color="#383835" if is_dark() else "#c3c2b7", width=1))


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, theme="streamlit", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
