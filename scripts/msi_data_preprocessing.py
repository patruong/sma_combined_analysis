import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================
# Utility Functions
# =====================

def create_spot_id(df):
    """Adds a unique spot ID to each row."""
    df = df.copy()
    df['spot_id'] = ['pixel_' + str(i) for i in df.index]
    return df

def create_intensities_df(df):
    """Creates a transposed dataframe of intensities with m/z values as columns."""
    mz_columns = [col for col in df.columns if col not in ['x', 'y']]
    df = create_spot_id(df)
    intensities_df = df[mz_columns].transpose()
    intensities_df.columns = df['spot_id']
    intensities_df.index = ['mz-' + str(col) for col in intensities_df.index]
    intensities_df = intensities_df.T.reset_index()
    return intensities_df

def create_metadata_df(df):
    """Creates a metadata dataframe with x, y coordinates and spot_id."""
    df = create_spot_id(df)
    metadata_df = df[['x', 'y', 'spot_id']]
    return metadata_df

def filter_zero_pixels(df):
    """Removes pixels where all m/z intensities are zero."""
    return df.loc[~(df.drop(columns=["x", "y"]) == 0).all(axis=1)]

def rotate_and_flip_coordinates(df, rotation=0, flip_x=False, flip_y=False):
    """
    Rotates and/or flips spatial coordinates in the dataframe.
    Rotation must be one of [0, 90, 180, 270].
    """
    if rotation not in [0, 90, 180, 270]:
        raise ValueError("Rotation must be one of [0, 90, 180, 270]")

    x_max = df['x'].max() + 1
    y_max = df['y'].max() + 1
    value_cols = df.columns.difference(['x', 'y'])

    grid = np.full((y_max, x_max, len(value_cols)), np.nan)
    for _, row in df.iterrows():
        grid[int(row['y']), int(row['x']), :] = row[value_cols].values

    # Apply rotation
    k = {0: 0, 90: 3, 180: 2, 270: 1}[rotation]
    transformed = np.rot90(grid, k=k)

    # Apply flipping
    if flip_y:
        transformed = np.flipud(transformed)
    if flip_x:
        transformed = np.fliplr(transformed)

    # Reconstruct DataFrame
    new_y, new_x = transformed.shape[:2]
    coords = [(x, y) for y in range(new_y) for x in range(new_x)]
    reshaped = transformed.reshape(-1, transformed.shape[2])
    new_df = pd.DataFrame(reshaped, columns=value_cols)
    new_df.insert(0, 'y', [coord[1] for coord in coords])
    new_df.insert(0, 'x', [coord[0] for coord in coords])

    return new_df

# =====================
# Main Script
# =====================

def main():
    # File paths
    # input_path should be the downloaded files from sma.zip in https://data.mendeley.com/datasets/w7nw4km7xd/1
    input_path = "./sma/V11L12-038/V11L12-038_D1/output_data/V11L12-038_D1_MSI/V11L12-038_Mouse_D1.Visium.9aa.220826_smamsi.csv"
    # output_path should be the input path used by magpie.
    output_dir = "./input/V11L12-038_D1/msi"
    os.makedirs(output_dir, exist_ok=True)

    # Load and process data
    df = pd.read_csv(input_path)
    df = rotate_and_flip_coordinates(df, rotation=90, flip_x=False, flip_y=True)
    df_filtered = filter_zero_pixels(df)

    # Create and save outputs
    intensities = create_intensities_df(df_filtered)
    metadata = create_metadata_df(df_filtered)

    intensities.to_csv(f"{output_dir}/MSI_intensities.csv", index=False)
    metadata.to_csv(f"{output_dir}/MSI_metadata.csv", index=False)

    # Quick visualization
    plt.scatter(x=metadata['x'], y=metadata['y'], s=1, c='red')
    plt.title("MSI Spot Coordinates")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    main()
