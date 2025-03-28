import pandas as pd
import matplotlib.pyplot as plt

def visualize_linear_models_estimates(result_path):
    # Load results
    df = pd.read_csv(result_path)

    # Round numeric values before formatting
    df["RMSE"] = df["RMSE"].round(2)
    if pd.api.types.is_numeric_dtype(df["Alpha"]):
        df["Alpha"] = df["Alpha"].round(4)
    if pd.api.types.is_numeric_dtype(df["L1_Ratio"]):
        df["L1_Ratio"] = df["L1_Ratio"].round(2)

    # Replace NaNs with em dash
    df.fillna("–", inplace=True)

    # Rename column names to match the lecture slides
    df.rename(columns={"Alpha": "λ", "L1_Ratio": "α"}, inplace=True)

    # Identify row with lowest RMSE
    best_row_idx = df["RMSE"].astype(float).idxmin() + 1  # +1 because row 0 is header in matplotlib table

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Apply styles to cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9d9d9")  # header
        elif col == 0:
            cell.set_facecolor("#cfe2f3")  # model name column
        if row == best_row_idx:
            cell.set_facecolor("#ffefb0")  # soft gold background

    # Save and display
    plt.savefig(f"results/linearmodels/model_comparison_table.png", bbox_inches='tight', dpi=300)
    plt.show()
