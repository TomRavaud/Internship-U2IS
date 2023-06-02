import numpy as np
from tabulate import tabulate
import traversalcost.time_features
import inspect


def get_features(roll_rate_values,
                 pitch_rate_values,
                 vertical_acceleration_values,
                 dict):
    """Extract features from IMU signals and concatenate them

    Args:
        roll_rate_values (list): List of roll rate values
        pitch_rate_values (list): List of pitch rate values
        vertical_acceleration_values (list): List of vertical acceleration
        values
        dict (dictionary): Dictionary containing the function to be applied to
        each signal and the parameters to be passed to the function

    Returns:
        ndarray (n,): Concatenated features
    """
    # Compute the features from the IMU signals
    features = [
        dict["function"](roll_rate_values,
                         **dict["params_roll_rate"]),
        dict["function"](pitch_rate_values,
                         **dict["params_pitch_rate"]),
        dict["function"](vertical_acceleration_values,
                         **dict["params_vertical_acceleration"])
    ]
    
    # Concatenate the features
    features = np.concatenate(features)
    
    return features

def generate_description(dict):
    """Generate a description of a function used to extract features from IMU
    signals

    Args:
        dict (dictionary): Dictionary containing the function to be applied to
        each signal and the parameters to be passed to the function

    Returns:
        table: A table containing the description of the function
    """
    # Generate a dummy signal and extract the features
    dummy_signal = np.random.rand(100)
    dummy_features = dict["function"](dummy_signal)
    
    # Get the default arguments of the function
    signature = inspect.signature(dict["function"])
    default_args = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    
    # Copy and update the default arguments with the roll rate parameters
    args_roll_rate = default_args.copy()
    args_roll_rate.update(dict["params_roll_rate"])
    
    # Copy and update the default arguments with the pitch rate parameters
    args_pitch_rate = default_args.copy()
    args_pitch_rate.update(dict["params_pitch_rate"])
    
    # Copy and update the default arguments with the vertical acceleration
    # parameters
    args_vertical_acceleration = default_args.copy()
    args_vertical_acceleration.update(dict["params_vertical_acceleration"])
    
    # Generate the description of the function
    data = [
        [
            "Function",
            "Description",
            "Nb features/signal",
            "Params roll rate",
            "Params pitch rate",
            "Params vertical acceleration",
        ],
        [
            dict["function"].__name__,
            (dict["function"].__doc__).split("\n")[0],
            len(dummy_features),
            args_roll_rate,
            args_pitch_rate,
            args_vertical_acceleration,
        ],
    ]
    
    # Generate the table
    table = tabulate(data,
                     headers="firstrow",
                     tablefmt="fancy_grid",
                     maxcolwidths=20,
                     numalign="center",)
    
    return table


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    # Test the function
    FEATURES = {"function": traversalcost.time_features.variance,
                "params_roll_rate": {},
                "params_pitch_rate": {},
                "params_vertical_acceleration": {}}
    
    print(generate_description(FEATURES))
