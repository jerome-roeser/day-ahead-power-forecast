import matplotlib.pyplot as plt
import pandas as pd


def compress(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Reduces size of dataframe by downcasting numerical columns
    """
    input_size = df.memory_usage(index=True).sum() / 1024
    print("dataframe size: ", round(input_size, 2), "kB")

    in_size = df.memory_usage(index=True).sum()
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100

    print("optimized size by {} %".format(round(ratio, 2)))
    print("new dataframe size: ", round(out_size / 1024, 2), " kB")

    return df


def plot_loss_mae(history):
    # Setting figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Create the plots
    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])

    ax2.plot(history.history["mae"])
    ax2.plot(history.history["val_mae"])

    # Set titles and labels
    ax1.set_title("Model loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")

    ax2.set_title("Mean Absolute Error")
    ax2.set_ylabel("MAE")
    ax2.set_xlabel("Epoch")

    # Set limits for y-axes
    # ax1.set_ylim(ymin=0, ymax=20)
    # ax2.set_ylim(ymin=0, ymax=200)

    # Generate legends
    ax1.legend(["Train", "Validation"], loc="best")
    ax2.legend(["Train", "Validation"], loc="best")

    # Show grids
    ax1.grid(axis="x", linewidth=0.5)
    ax1.grid(axis="y", linewidth=0.5)

    ax2.grid(axis="x", linewidth=0.5)
    ax2.grid(axis="y", linewidth=0.5)

    plt.show()
