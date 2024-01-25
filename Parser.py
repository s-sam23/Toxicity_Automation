import argparse

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, cmd_line=None):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser.add_argument('--data', default='Data', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--Models', type=str, default='Model_Weights', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--Results', type=str, default='Results', help='Interpretable name')
        parser.add_argument('--temp', type=str, default='temp', help='temporary files')  
        parser.add_argument('--Validation', type=str, default='Validation_data', help='temporary files')
        parser.add_argument('--Validation_location', type=str, default='Validation_folder', help='temporary files')
        parser.add_argument('--Prediction', type=str, default='Prediction_data', help='temporary files')         
        parser.add_argument('--Prediction_location', type=str, default='Prediction_folder', help='temporary files')
        self.initialized = True
        return parser