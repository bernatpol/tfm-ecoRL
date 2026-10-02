import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def plot_with_rolling_mean_and_regression(
    df, x, y, window_size=10, title="Title of the graph", xlabel=None, ylabel=None,
    plot_size=(14, 7), grid=True, regression_line=True, alpha=0.3, rolling_mean_alpha=0.4,
    rolling_mean_color="blue", regression_line_style="--", regression_line_color="orange", ax=None
):
    """
    Plots a graph with the original data, rolling mean, and an optional regression line.

    Parameters:
    ------------
    df : DataFrame
        DataFrame containing the data to be plotted.

    x : str
        The column name for the x-axis data.

    y : str
        The column name for the y-axis data.

    window_size : int, optional, default=10
        The window size for calculating the rolling mean.

    title : str, optional, default="Title of the graph"
        Title for the plot.

    xlabel : str, optional
        Label for the x-axis. Defaults to the name of the x column.

    ylabel : str, optional
        Label for the y-axis. Defaults to the name of the y column.

    plot_size : tuple, optional, default=(14, 7)
        Size of the plot in inches (width, height).

    grid : bool, optional, default=True
        Whether to display the grid on the plot.

    regression_line : bool, optional, default=True
        Whether to include a linear regression line.

    alpha : float, optional, default=0.3
        Transparency for the original data points.

    rolling_mean_alpha : float, optional, default=0.4
        Transparency for the rolling mean line.

    rolling_mean_color : str, optional, default="blue"
        Color of the rolling mean line.

    regression_line_style : str, optional, default="--"
        Line style for the regression line.

    regression_line_color : str, optional, default="orange"
        Color of the regression line.
    """
    # Drop NaN values for x and y
    df = df.dropna(subset=[x, y])

    # Sort the DataFrame by the x column
    df = df.sort_values(by=x)

    # Calculate the rolling mean with the specified window size
    df["Rolling_Mean"] = df[y].rolling(window=window_size).mean()

    # Prepare for regression
    x_regression = np.array(pd.to_numeric(df[x]))
    y_regression = np.array(pd.to_numeric(df[y]))
    mask = ~np.isnan(x_regression) & ~np.isnan(y_regression)  # Remove NaN values for regression
    x_regression = x_regression[mask]
    y_regression = y_regression[mask]

    # Perform linear regression
    if regression_line:
        slope, intercept = np.polyfit(x_regression, y_regression, 1)
        regression_line_values = slope * x_regression + intercept

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=plot_size)
    else:
        fig = ax.get_figure()

    # Plot the original data points with transparency
    ax.plot(df[x], df[y], "-", alpha=alpha, label="Original Data")

    # Plot the rolling mean
    ax.plot(
        df[x],
        df["Rolling_Mean"],
        label=f"Rolling Mean (window={window_size})",
        color=rolling_mean_color,
        alpha=rolling_mean_alpha,
        linewidth=2,
    )

    # Plot the regression line if enabled
    if regression_line:
        ax.plot(
            df[x],
            regression_line_values,
            label="Regression Line",
            linestyle=regression_line_style,
            color=regression_line_color,
            linewidth=2,
        )

    # Customize labels and title
    ax.set_xlabel(xlabel if xlabel else x)
    ax.set_ylabel(ylabel if ylabel else y)
    ax.set_title(title, fontweight="bold", fontsize=16)

    # Add a legend
    ax.legend()

    # Show the grid if specified
    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.2)

    # Display the plot
    return ax


def plot_decision_trajectory(xy, t, norm, action=None, reward=None, cmap = plt.cm.viridis, title="Decision trajectory"):
    plt.figure(figsize=(6, 6))

    # --- Gradient line (Option 2) ---
    for i in range(len(xy) - 1):
        plt.plot(
            xy[i:i+2, 0],
            xy[i:i+2, 1],
            color=cmap(norm(t[i])),
            linewidth=2
        )

    # --- Reward points on top (Option 1) ---
    if reward is not None:
        for i in range(len(xy)):
            if reward[i] > 0:
                plt.scatter(
                    xy[i, 0],
                    xy[i, 1],
                    color='gold',
                    s=100,
                    edgecolor='black',
                    linewidth=1,
                    marker='o'
                )

    # --- Scatter on top (Option 1) ---
    if action is None:
        sc = plt.scatter(
            xy[:, 0],
            xy[:, 1],
            c=t,
            cmap=cmap,
            s=20,
            edgecolor='none'
        )
    else:
        # Draw invisible points to create the colorbar
        sc = plt.scatter(
            xy[:, 0],
            xy[:, 1],
            c=t,
            cmap=cmap,
            s=0,
            edgecolor='none'
        )
        for i in range(len(xy)):
            label = 'L' if action[i] == 0 else 'R'
            plt.text(
                xy[i, 0],
                xy[i, 1],
                label,
                fontsize=15 if i < 20 else 15 - 5/len(xy),
                ha='center',
                va='center',
                color='black'
            )

    # --- Start and End points ---
    plt.scatter(xy[0, 0], xy[0, 1], color='red', s=20, label='Start', zorder=3, marker='*')

    # --- Colorbar ---
    plt.colorbar(sc, label="Time step")

    # --- Labels ---
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    ax = plt.gca()
    return ax