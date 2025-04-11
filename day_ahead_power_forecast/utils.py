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

    print(f"optimized size by {ratio:.2f} %")
    print(f"new dataframe size: {out_size / 1024:.2f} kB")

    return df


def plot_loss_mae(history: dict) -> None:
    """
    Side by side plot of loss and mae training metrics

    Parameters
    ----------
    history (dict)
        History dict returned by the training of the pytorch model
    """
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

    # Generate legends
    ax1.legend(["Train", "Validation"], loc="best")
    ax2.legend(["Train", "Validation"], loc="best")

    # Show grids
    ax1.grid(axis="x", linewidth=0.5)
    ax1.grid(axis="y", linewidth=0.5)

    ax2.grid(axis="x", linewidth=0.5)
    ax2.grid(axis="y", linewidth=0.5)

    plt.show()
