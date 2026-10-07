class Workflow():
    def __init__(self):
        # TODO
        pass

    def run(self):
        # TODO
        pass

    def print(self):
        # TODO
        pass

class CompilerApi():
    def __init__(self):
        # TODO
        pass

    def compile_workflow(self, model, treatmentVar, treatmentVarValues, datasetPath):
        # TODO:
        # - only allows 1 treatment var... allow multiple
        # - allow other model types beyond OLS
        """
        Compile the workflow

        Parameters
        ----------
        model : object
            The OLS model, which is a RegressionResultsWrapper object.
            Ultimately just care about the model's weights.
        treatmentVar : int
            The index position of the treatment variable.
        treatmentVarValues : list
            The values for each treatment variable (ex [0, 1]).
        datasetPath : str
            The path to the dataset to be used for compilation.

        Returns
        -------
        compiled_model : object
            NOTE: Unsure what specifically should be returned...
        """
        pass

    def _extractWeights(self):
        # TODO
        pass

    def _emitMLIR(self):
        # TODO
        pass

    def _buildWorkflowObject(self):
        # TODO
        pass