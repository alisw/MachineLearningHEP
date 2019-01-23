#############################################################################
##  © Copyright CERN 2018. All rights not expressly granted are reserved.  ##
##                 Author: Gian.Michele.Innocenti@cern.ch                  ##
## This program is free software: you can redistribute it and/or modify it ##
##  under the terms of the GNU General Public License as published by the  ##
## Free Software Foundation, either version 3 of the License, or (at your  ##
## option) any later version. This program is distributed in the hope that ##
##  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  ##
##     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    ##
##           See the GNU General Public License for more details.          ##
##    You should have received a copy of the GNU General Public License    ##
##   along with this program. if not, see <https://www.gnu.org/licenses/>. ##
#############################################################################

"""
Methods to: configure parameters
"""

import os.path
from pkg_resources import resource_stream
import yaml
from machine_learning_hep.logger import get_logger
from machine_learning_hep.io import dump_yaml_from_dict, parse_yaml, print_dict


class Configuration:

    config_paths = {"run": "config_run_parameters.yml", "models": "config_model_parameters.yml"}

    def __init__(self, run_config_input=None, model_config_input=None):
        self.run_config_input = run_config_input
        self.model_config_input = model_config_input

        self.run_config = {}
        self.model_config = {}

    def run_config_source(self, source=None):
        self.run_config_input = source

    def model_config_source(self, source=None):
        self.model_config_input = source

    @staticmethod
    def get_meta_config(which_config):
        """
        Load a default config
        Args:
            which_config: either "run" or "models"
        """
        if which_config not in Configuration.config_paths:
            get_logger().critical("Unknown configuration %s. Cannot be loaded.", which_config)
        stream = resource_stream("machine_learning_hep.data",
                                 Configuration.config_paths[which_config])
        return yaml.safe_load(stream)

    @staticmethod
    def construct_default_model_config():
        config = Configuration.get_meta_config("models")
        #default_config = {}
        for mltype in list(config.keys()):
            for backend in list(config[mltype]):
                for model in list(config[mltype][backend]):
                    config[mltype][backend][model] = config[mltype][backend][model]["default"]
        return config

    @staticmethod
    def construct_default_run_config():
        config = Configuration.get_meta_config("run")

        config = {k: config[k]["default"] for k in config}
        model_config = Configuration.get_meta_config("models")

        # Add mltype, backends and corresponding models from model configuration
        for mltype, backends in model_config.items():
            config["activate_models"][mltype] = {}
            for backend, impl in backends.items():
                config["activate_models"][mltype][backend] = {}
                for model, parameters in impl.items():
                    config["activate_models"][mltype][backend][model] = {}

        # Fill run_config with default parameters from default model configuration
        for mltype, backends in model_config.items():
            for backend, impl in backends.items():
                for model, parameters in impl.items():
                    config["activate_models"][mltype][backend][model] = parameters["activate"]

        return config

    @staticmethod
    def dump_default_config(which_config, path):
        """
        Write default configuration
        Args:
            which_config: either "run" or "models"
            path: full or relative path to where the config should be dumped
        """
        construction_functions = {"run": Configuration.construct_default_run_config,
                                  "models": Configuration.construct_default_model_config}
        if which_config not in construction_functions:
            get_logger().critical("No defaults for %s.", which_config)
        path = os.path.expanduser(path)
        dump_yaml_from_dict(construction_functions[which_config](), path)
        get_logger().info("Dumped default %s config to %s", which_config, path)


    def assert_run_config(self):
        """
        Validate and return the configuration for run
        Args:
            path: path to configuration YAML
        """
        logger = get_logger()
        logger.debug("Check sanity of user configs")


        user_run_config = {}
        if isinstance(self.run_config_input, str):
            user_run_config = parse_yaml(os.path.expanduser(self.run_config_input))
        elif isinstance(self.run_config_input, dict):
            user_run_config = self.run_config_input


        # At this point the asserted_config dict is just the one with defaults
        run_config = Configuration.get_meta_config("run")
        asserted_config = {k: run_config[k]["default"] for k in run_config}
        choices_config = {k: run_config[k]["choices"]
                          for k in run_config if "choices" in run_config[k]}
        depends_config = {k: run_config[k]["depends"]
                          for k in run_config if "depends" in run_config[k]}
        types_config = {k: run_config[k]["type_as"]
                        for k in run_config if "type_as" in run_config[k]}
        # Check for unknown parameters and abort since running entire machinery with wrong
        # setting (e.g. 'dotaining' instead of 'dotraining' might happen just by accident)
        # could be just overhead.
        for k in user_run_config:
            if k not in asserted_config:
                logger.critical("Unkown parameter %s in config", k)
            elif user_run_config[k] is None:
                logger.critical("Missing value for parameter %s in config", k)

        # Replace all defaults if user specified parameter
        for k in asserted_config:
            asserted_config[k] = user_run_config.get(k, asserted_config[k])
            # If parameter is already set, check if consistent
            if k in choices_config and asserted_config[k] not in choices_config[k]:
                logger.critical("Invalid value %s for parameter %s. Must be one of %s",
                                str(user_run_config[k]), k, str(choices_config[k]))
            if k in types_config:
                check_types = [type(t) for t in types_config[k]]
                if not isinstance(asserted_config[k], tuple(check_types)):
                    logger.critical("Invalid value type %s of parameter %s. Must be of type %s",
                                    str(type(asserted_config[k])), k, str(check_types))

        # Can so far only depend on one parameter, change to combination
        # of parameters. Do we need to check for circular dependencies?
        for k in depends_config:
            if (asserted_config[depends_config[k]["parameter"]]
                    == depends_config[k]["value"]
                    and asserted_config[k] != depends_config[k]["set"]):
                asserted_config[k] = depends_config[k]["set"]
                logger.info("Parameter %s = %s enforced since it is required for %s == %s",
                            k, str(depends_config[k]["set"]),
                            str(depends_config[k]["parameter"]),
                            str(depends_config[k]["value"]))

        self.run_config = asserted_config


    def assert_model_config(self): # pylint: disable=R0912
        """
        Validate and return the configuration for ml models
        Args:
            path: path to configuration YAML
            run_config: Run configuration since loading some models can depend on that, e.g.
                        if run_config["activate_keras"] == 0 the keras config does not need
                        to be checked and loaded.
        """
        logger = get_logger()
        logger.debug("Check sanity of user configs")

        user_config = {}
        if isinstance(self.model_config_input, str):
            user_config = parse_yaml(os.path.expanduser(self.model_config_input))
        elif isinstance(self.model_config_input, dict):
            user_config = self.model_config_input

        # At this point the asserted_config dict is just the one with defaults
        asserted_config = Configuration.get_meta_config("models")[self.run_config["mltype"]]
        user_config = user_config.get(self.run_config["mltype"], {})

        # Could probably merged with the former loop, however, would like to see whether there are
        # e.g. typos. Because steering a run wanting keras - but writing kras - could cost a lot of
        # time when it needs to be done again.
        if self.run_config["mltype"] in self.run_config["activate_models"]:
            for backend, model in \
            self.run_config["activate_models"][self.run_config["mltype"]].items():
                if backend not in asserted_config:
                    logger.critical("Unknown backend %s.", backend)
                if model is None:
                    logger.critical("No models specified for backend %s.", backend)
                for name, activate in model.items():
                    if name not in asserted_config[backend]:
                        logger.critical("Unknown model %s for backend %s.", name, backend)
                    if name in asserted_config[backend]:
                        if activate is None or not isinstance(activate, bool):
                            logger.critical("Activation value of model %s for backend %s " \
                                             "must be specified as boolean value.", name, backend)
                        asserted_config[backend][name]["activate"] = activate



        # Pop deactivated models
        for backend in list(asserted_config.keys()):
            for model in list(asserted_config[backend].keys()):
                if not asserted_config[backend][model]["activate"]:
                    del asserted_config[backend][model]
                else:
                    asserted_config[backend][model] = asserted_config[backend][model]["default"]
                    if backend in user_config and model in user_config[backend]:
                        if len(user_config[backend][model]) != len(asserted_config[backend][model]):
                            logger.critical("Parameter list for %s model %s differs",
                                            backend, model)
                        for u in asserted_config[backend][model]:
                            asserted_config[backend][model][u] = \
                                user_config[backend][model].get(u,
                                                                asserted_config[backend][model][u])

        self.model_config = asserted_config



    def configure(self):
        self.assert_run_config()
        self.assert_model_config()


    def get_run_config(self):
        return self.run_config


    def get_model_config(self):
        return self.model_config


    def print_configuration(self, run_config_flag=True, model_config_flag=True):
        logger = get_logger()
        logger.info("Run and model configuration:")
        if run_config_flag:
            print_dict(self.run_config, skip=["activate_models"])
        if model_config_flag:
            print_dict(self.model_config)
        print("====================================================")
