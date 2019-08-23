class Config(object):

    def __init__(self, config, config_type):
        self._config = config.get(config_type)

    def get_property(self, property_name):
        return self._config.get(property_name)


class OptimizerConfig(Config):

    @property
    def nn_filename(self):
        return self.get_property('nn_filename')

    @nn_filename.setter
    def nn_filename(self, nn_filename):
        self._nn_filename = nn_filename

    @property
    def constants_filename(self):
        return self.get_property('constants_filename')

    @constants_filename.setter
    def constants_filename(self, constants_filename):
        self._constants_filename = constants_filename

    @property
    def training_data_filename(self):
        return self.get_property('training_data_filename')

    @training_data_filename.setter
    def training_data_filename(self, training_data_filename):
        self._training_data_filename = training_data_filename

    @property
    def output_filename(self):
        return self.get_property('output_filename')

    @output_filename.setter
    def output_filename(self, output_filename):
        self._output_filename = output_filename
        
    @property
    def maximum_number_increasable_node(self):
        return self.get_property('maximum_number_increasable_node')

    @maximum_number_increasable_node.setter
    def maximum_number_increasable_node(self, maximum_number_increasable_node):
        self.maximum_number_increasable_node = maximum_number_increasable_node
    
    @property
    def minimum_number_reducible_node(self):
        return self.get_property('minimum_number_reducible_node')

    @minimum_number_reducible_node.setter
    def minimum_number_reducible_node(self, minimum_number_reducible_node):
        self.minimum_number_reducible_node = minimum_number_reducible_node