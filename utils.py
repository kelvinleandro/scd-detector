import numpy as np


def sample_patient_data(X, y, pids, n, random_seed=None):
    # Set the random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Get unique patient IDs
    unique_pids = np.unique(pids)

    # Initialize lists to store the sampled data
    sampled_X = []
    sampled_y = []
    sampled_pids = []

    # Iterate over each unique patient ID
    for pid in unique_pids:
        # Find indices where the patient ID matches
        indices = np.where(pids == pid)[0]

        # If the patient has more than n samples, randomly choose n samples
        if len(indices) >= n:
            selected_indices = np.random.choice(indices, n, replace=False)
        else:
            # If the patient has fewer than n samples, take all available samples
            selected_indices = indices

        # Append the selected samples to the lists
        sampled_X.append(X[selected_indices])
        sampled_y.append(y[selected_indices])
        sampled_pids.append(pids[selected_indices])

    # Concatenate the lists to form the final arrays
    sampled_X = np.concatenate(sampled_X, axis=0)
    sampled_y = np.concatenate(sampled_y, axis=0)
    sampled_pids = np.concatenate(sampled_pids, axis=0)

    # Shuffle the data while maintaining alignment between X, y, and pids
    shuffle_indices = np.arange(len(sampled_X))
    np.random.shuffle(shuffle_indices)

    sampled_X = sampled_X[shuffle_indices]
    sampled_y = sampled_y[shuffle_indices]
    sampled_pids = sampled_pids[shuffle_indices]

    return sampled_X, sampled_y, sampled_pids
