import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def violin_plot_robustness(
    results_noise_normal_training: pd.DataFrame,
    results_noise_adversarial_training: pd.DataFrame,
    results_pgd_normal_training: pd.DataFrame,
    results_pgd_adversarial_training: pd.DataFrame,
    metric: str = "TARS",
    box_visible: bool = False,
):
    # add attack name to dataframe (only required for plotting)
    results_pgd_normal_training["attack"] = list(
        np.repeat("Untargeted PGD Attack", len(results_pgd_normal_training))
    )
    results_pgd_adversarial_training["attack"] = list(
        np.repeat("Untargeted PGD Attack", len(results_pgd_adversarial_training))
    )
    results_noise_normal_training["attack"] = list(
        np.repeat("Noise Attack", len(results_noise_normal_training))
    )
    results_noise_adversarial_training["attack"] = list(
        np.repeat("Noise Attack", len(results_noise_adversarial_training))
    )

    # add training name to dataframe (only required for plotting)
    results_pgd_normal_training["training"] = list(
        np.repeat("Normal Training", len(results_pgd_normal_training))
    )
    results_pgd_adversarial_training["training"] = list(
        np.repeat("Adversarial Training", len(results_pgd_adversarial_training))
    )
    results_noise_normal_training["training"] = list(
        np.repeat("Normal Training", len(results_noise_normal_training))
    )
    results_noise_adversarial_training["training"] = list(
        np.repeat("Adversarial Training", len(results_noise_adversarial_training))
    )

    # concatenate datasets with normal training (only required for plotting)
    results_normal_training = pd.concat(
        [results_pgd_normal_training, results_noise_normal_training]
    )

    # concatenate datasets with adversarial training (only required for plotting)
    results_adversarial_training = pd.concat(
        [results_pgd_adversarial_training, results_noise_adversarial_training]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Violin(
            x=results_normal_training["attack"],
            y=results_normal_training[metric],
            legendgroup="Normal Training",
            scalegroup="Normal Training",
            name="Normal Training",
            line_color="orange",
        )
    )
    fig.add_trace(
        go.Violin(
            x=results_adversarial_training["attack"],
            y=results_adversarial_training[metric],
            legendgroup="Adversarial Training",
            scalegroup="Adversarial Training",
            name="Adversarial Training",
            line_color="blue",
        )
    )

    fig.update_traces(box_visible=box_visible, meanline_visible=True, spanmode="hard")
    fig.update_layout(
        violinmode="group",
        # iolingap=0,
        # iolingroupgap = 0.1,
        yaxis_title=metric,
        legend=dict(font=dict(size=15)),
    )
    fig.update_xaxes(tickfont_size=15)
    fig.update_yaxes(titlefont=dict(size=15))

    return fig


def box_plot_robustness(
    results_noise_normal_training: pd.DataFrame,
    results_noise_adversarial_training: pd.DataFrame,
    results_pgd_normal_training: pd.DataFrame,
    results_pgd_adversarial_training: pd.DataFrame,
    metric: str = "TARS",
):
    # add attack name to dataframe (only required for plotting)
    results_pgd_normal_training["attack"] = list(
        np.repeat("Untargeted PGD Attack", len(results_pgd_normal_training))
    )
    results_pgd_adversarial_training["attack"] = list(
        np.repeat("Untargeted PGD Attack", len(results_pgd_adversarial_training))
    )
    results_noise_normal_training["attack"] = list(
        np.repeat("Noise Attack", len(results_noise_normal_training))
    )
    results_noise_adversarial_training["attack"] = list(
        np.repeat("Noise Attack", len(results_noise_adversarial_training))
    )

    # add training name to dataframe (only required for plotting)
    results_pgd_normal_training["training"] = list(
        np.repeat("Normal Training", len(results_pgd_normal_training))
    )
    results_pgd_adversarial_training["training"] = list(
        np.repeat("Adversarial Training", len(results_pgd_adversarial_training))
    )
    results_noise_normal_training["training"] = list(
        np.repeat("Normal Training", len(results_noise_normal_training))
    )
    results_noise_adversarial_training["training"] = list(
        np.repeat("Adversarial Training", len(results_noise_adversarial_training))
    )

    # concatenate datasets with normal training (only required for plotting)
    results_normal_training = pd.concat(
        [results_pgd_normal_training, results_noise_normal_training]
    )

    # concatenate datasets with adversarial training (only required for plotting)
    results_adversarial_training = pd.concat(
        [results_pgd_adversarial_training, results_noise_adversarial_training]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            x=results_normal_training["attack"],
            y=results_normal_training[metric],
            name="Normal Training",
            marker_color="orange",
        )
    )
    fig.add_trace(
        go.Box(
            x=results_adversarial_training["attack"],
            y=results_adversarial_training[metric],
            name="Adversarial Training",
            marker_color="blue",
        )
    )

    # fig.update_traces(boxpoints= False)
    fig.update_layout(
        boxmode="group",
        # boxgap=0,
        # boxgroupgap = 0.1,
        width=900,
        height=500,
        yaxis_title=metric,
        legend=dict(font=dict(size=15)),
    )
    fig.update_xaxes(tickfont_size=15)
    fig.update_yaxes(titlefont=dict(size=15))

    return fig


def box_plot_robustness_targeted(
    results_decreasing_normal_training: pd.DataFrame,
    results_decreasing_adversarial_training: pd.DataFrame,
    results_increasing_normal_training: pd.DataFrame,
    results_increasing_adversarial_training: pd.DataFrame,
    results_constant_normal_training: pd.DataFrame,
    results_constant_adversarial_training: pd.DataFrame,
    results_zigzag_normal_training: pd.DataFrame,
    results_zigzag_adversarial_training: pd.DataFrame,
    metric: str = "TARS",
):
    # add attack name to dataframe (only required for plotting)
    results_increasing_normal_training["attacker's target"] = list(
        np.repeat("increasing", len(results_increasing_normal_training))
    )
    results_increasing_adversarial_training["attacker's target"] = list(
        np.repeat("increasing", len(results_increasing_adversarial_training))
    )
    results_decreasing_normal_training["attacker's target"] = list(
        np.repeat("decreasing", len(results_decreasing_normal_training))
    )
    results_decreasing_adversarial_training["attacker's target"] = list(
        np.repeat("decreasing", len(results_decreasing_adversarial_training))
    )
    results_constant_normal_training["attacker's target"] = list(
        np.repeat("constant", len(results_constant_normal_training))
    )
    results_constant_adversarial_training["attacker's target"] = list(
        np.repeat("constant", len(results_constant_adversarial_training))
    )
    results_zigzag_normal_training["attacker's target"] = list(
        np.repeat("zigzag", len(results_zigzag_normal_training))
    )
    results_zigzag_adversarial_training["attacker's target"] = list(
        np.repeat("zigzag", len(results_zigzag_adversarial_training))
    )

    # add training name to dataframe (only required for plotting)
    results_increasing_normal_training["training"] = list(
        np.repeat("Normal Training", len(results_increasing_normal_training))
    )
    results_increasing_adversarial_training["training"] = list(
        np.repeat("Adversarial Training", len(results_increasing_adversarial_training))
    )
    results_decreasing_normal_training["training"] = list(
        np.repeat("Normal Training", len(results_decreasing_normal_training))
    )
    results_decreasing_adversarial_training["training"] = list(
        np.repeat("Adversarial Training", len(results_decreasing_adversarial_training))
    )
    results_constant_normal_training["training"] = list(
        np.repeat("Normal Training", len(results_constant_normal_training))
    )
    results_constant_adversarial_training["training"] = list(
        np.repeat("Adversarial Training", len(results_constant_adversarial_training))
    )
    results_zigzag_normal_training["training"] = list(
        np.repeat("Normal Training", len(results_zigzag_normal_training))
    )
    results_zigzag_adversarial_training["training"] = list(
        np.repeat("Adversarial Training", len(results_zigzag_adversarial_training))
    )

    # concatenate datasets with normal training (only required for plotting)
    results_normal_training = pd.concat(
        [
            results_increasing_normal_training,
            results_decreasing_normal_training,
            results_constant_normal_training,
            results_zigzag_normal_training,
        ]
    )

    # concatenate datasets with adversarial training (only required for plotting)
    results_adversarial_training = pd.concat(
        [
            results_increasing_adversarial_training,
            results_decreasing_adversarial_training,
            results_constant_adversarial_training,
            results_zigzag_adversarial_training,
        ]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            x=results_normal_training["attacker's target"],
            y=results_normal_training[metric],
            name="ordinary training",
            marker_color="orange",
            marker_opacity=0.1,
        )
    )
    fig.add_trace(
        go.Box(
            x=results_adversarial_training["attacker's target"],
            y=results_adversarial_training[metric],
            name="adversarial training",
            marker_color="blue",
            marker_opacity=0.1,
        )
    )

    # fig.update_traces(boxpoints= False)
    fig.update_layout(
        boxmode="group",
        # boxgap=0,
        # boxgroupgap = 0.1,
        width=1500,
        height=500,
        yaxis_title=metric,
        xaxis_title="attacker's target",
        legend=dict(font=dict(size=25)),
        legend_title=dict(font=dict(size=25)),
    )

    fig.update_yaxes(titlefont=dict(size=25), tickfont=dict(size=20))
    fig.update_xaxes(titlefont=dict(size=25), tickfont=dict(size=20))

    return fig


def line_plot(
    dataframe: pd.DataFrame,
    colors: list,
    dash: list,
    xaxis_title,
    yaxis_title,
    legend_title_text,
    width=None,
):
    layout = go.Layout(autosize=False, width=750, height=500)

    fig = go.Figure(layout=layout)

    for idx, col in enumerate(dataframe.columns):
        timesteps = list(dataframe.index.values + 1)
        y_values = dataframe[col]

        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=y_values,
                mode="lines",
                name=col,
                line=dict(color=colors[idx], dash=dash[idx], width=width),
            )
        )

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title_text=legend_title_text,
        legend=dict(font=dict(size=20)),
        legend_title=dict(font=dict(size=20)),
    )
    fig.update_xaxes(titlefont=dict(size=20), tickfont=dict(size=15))
    fig.update_yaxes(titlefont=dict(size=20), tickfont=dict(size=15))

    # only used for targeted attack illustration
    # fig.update_layout(legend=dict(orientation="h",
    #                              itemwidth=40,
    #                              yanchor="bottom",
    #                              y=1.02,
    #                              xanchor="right",
    #                              x=1))

    return fig


def line_fill_plot(
    dataframe: pd.DataFrame,
    colors: list,
    dash: list,
    fill: list,
    fillpattern_shape: list,
    showlegend: list,
    fillcolor: list,
    xaxis_title,
    yaxis_title,
    legend_title_text,
    dtick=None,
):
    layout = go.Layout(
        autosize=False, width=750, height=500, legend={"traceorder": "normal"}
    )

    fig = go.Figure(layout=layout)

    for idx, col in enumerate(dataframe.columns):
        timesteps = list(dataframe.index.values + 1)
        y_values = dataframe[col]

        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=y_values,
                mode="lines",
                name=col,
                line=dict(color=colors[idx], dash=dash[idx]),
                fill=fill[idx],
                fillcolor=fillcolor[idx],
                fillpattern_shape=fillpattern_shape[idx],
                showlegend=showlegend[idx],
            )
        )

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title_text=legend_title_text,
        legend=dict(font=dict(size=20)),
        legend_title=dict(font=dict(size=20)),
    )
    fig.update_xaxes(titlefont=dict(size=20), tickfont=dict(size=15))

    if dtick:
        fig.update_yaxes(
            titlefont=dict(size=20), tickfont=dict(size=15), tick0=0.0, dtick=dtick
        )
    else:
        fig.update_yaxes(titlefont=dict(size=20), tickfont=dict(size=15))

    return fig


def heatmap_plot(
    data: np.ndarray,  # 3D array (channels, height, width)
    column_titles: list,
    yaxis_title: str,
    colorscale: str,
    zmin: float,
    zmax: float,
    width_adjustment: float,
):
    num_channels = data.shape[0]

    fig = make_subplots(
        rows=1,
        cols=num_channels,
        shared_xaxes=True,
        shared_yaxes=True,
        column_titles=column_titles,
        horizontal_spacing=0.1 / num_channels,
    )

    for channel in range(num_channels):
        fig.add_trace(
            go.Heatmap(
                z=data[channel],
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                colorbar=dict(tickfont=dict(size=30)),
            ),
            1,
            channel + 1,
        )

    fig.update_layout(
        height=500,
        width=((425 * num_channels) - width_adjustment),
        yaxis_title=yaxis_title,
    )

    fig.update_yaxes(
        titlefont=dict(size=30), tickfont=dict(size=25), autorange="reversed"
    )
    fig.update_xaxes(titlefont=dict(size=30), tickfont=dict(size=25))

    # font size of column titles
    fig.update_annotations(font_size=30)

    return fig


def line_subplot(
    data: list,
    colors: list,
    dash: list,
    column_titles: list,
    xaxis_title: str,
    yaxis_title: str,
    legend_title_text: str,
    width_adjustment: float,
    x_legend=None,
):
    num_plots = len(data)

    fig = make_subplots(
        rows=1,
        cols=num_plots,
        shared_xaxes=True,
        shared_yaxes=False,
        column_titles=column_titles,
        horizontal_spacing=0.15 / num_plots,
    )

    for plot in range(num_plots):
        dataframe = data[plot]
        for idx, col in enumerate(dataframe.columns):
            timesteps = list(dataframe.index.values + 1)
            y_values = dataframe[col]

            if plot == 0:
                fig.add_trace(
                    go.Scatter(
                        x=timesteps,
                        y=y_values,
                        mode="lines",
                        name=col,
                        line=dict(color=colors[idx], dash=dash[idx]),
                    ),
                    1,
                    plot + 1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=timesteps,
                        y=y_values,
                        mode="lines",
                        name=col,
                        line=dict(color=colors[idx], dash=dash[idx]),
                        showlegend=False,
                    ),
                    1,
                    plot + 1,
                )

    fig.update_layout(
        height=500,
        width=((750 * num_plots) - width_adjustment),
        yaxis_title=yaxis_title,
        legend_title_text=legend_title_text,
        legend=dict(font=dict(size=35)),
        legend_title=dict(font=dict(size=35)),
    )

    for plot in range(num_plots):
        fig["layout"][f"xaxis{plot+1}"]["title"] = xaxis_title

    fig.update_yaxes(titlefont=dict(size=35), tickfont=dict(size=30))
    fig.update_xaxes(titlefont=dict(size=35), tickfont=dict(size=30))

    # font size of column titles
    fig.update_annotations(font_size=35)

    # only used for targeted attack illustration
    fig.update_layout(
        legend=dict(
            orientation="h",
            itemwidth=35,
            yanchor="bottom",
            y=1.2,
            xanchor="right",
            x=x_legend,
        )
    )

    return fig
