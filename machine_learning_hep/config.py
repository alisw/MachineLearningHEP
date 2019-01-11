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
from copy import deepcopy


class Configuration:

    config_paths = {"run": "config_run_parameters.yml", "models": "config_model_parameters.yml"}

    def __init__(self, run_config_path=None, model_config_path=None):
        self.run_config_path = run_config_path
        self.model_config_path = model_config_path

        self.run_config = {}
        self.model_config = {}

    @staticmethod
    def get_meta_config(which_config):
        """
        Load a default config
        Args:
            which_config: either "run" or "models"
        """
        #config_paths = {"run": "config_run_parameters.yml", "models": "config_model_parameters.yml"}
        if which_config not in Configuration.config_paths:
            get_logger().critical("Unknown configuration %s. Cannot be loaded.", which_config)
        stream = resource_stream("machine_learning_hep.data", Configuration.config_paths[which_config])
        return yaml.safe_load(stream)

    @staticmethod
    def construct_default_model_config():
        config = Configuration.get_meta_config("models")
        default_config = {}
        for backend, impl in config.items():
            default_config[backend] = {}
            for model, parameters in impl.items():
                default_config[backend][model] = parameters["default"]
        return default_config

    @staticmethod
    def construct_default_run_config():
        config = Configuration.get_meta_config("run")

        config =  {k: config[k]["default"] for k in config}
        model_config = Configuration.get_meta_config("models")

        for backend, impl in model_config.items():
            for model, parameters in impl.items():
                if backend not in config["activate_models"]:
                    config["activate_models"][backend] = {}
                config["activate_models"][backend][model] = parameters["activate"]

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
        if self.run_config_path is not None:
            user_run_config = parse_yaml(self.run_config_path)


        # At this point the asserted_config dict is just the one with defaults
        run_config = Configuration.get_meta_config("run")
        asserted_config = {k: run_config[k]["default"] for k in run_config}
        choices_config = {k: run_config[k]["choices"] for k in run_config if "choices" in run_config[k]}
        depends_config = {k: run_config[k]["depends"] for k in run_config if "depends" in run_config[k]}
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
                if user_run_config[k] not in choices_config[k]:
                    logger.critical("Invalid value %s for parameter %s", str(user_run_config[k]), k)

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


    def assert_model_config(self):
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
        if self.model_config_path is not None:
            user_config = parse_yaml(self.model_config_path)

        # At this point the asserted_config dict is just the one with defaults
        asserted_config = Configuration.get_meta_config("models")


        # Remove everything which does not comply with the mltype
        tmp_config = {}
        for backend, impl in asserted_config.items():
            for model, parameters in impl.items():
                if backend not in tmp_config:
                    tmp_config[backend] = {}
                if parameters["mltype"] != self.run_config["mltype"]:
                    continue
                tmp_config[backend][model] = parameters

        # Could probably merged with the former loop, however, would like to see whether there are
        # e.g. typos. Because steering a run wanting keras - but writing kras - could cost a lot of
        # time when it needs to be done again.
        for backend, model in self.run_config["activate_models"].items():
            if backend not in asserted_config:
                logger.critical("Unknown backend %s.", backend)
            if model is None:
                logger.critical("No models specified for backend %s.", backend)
            for name, activate in model.items():
                if name not in asserted_config[backend]:
                    logger.critical("Unknown model %s for backend %s.", name, backend)
                if name in tmp_config[backend]:
                    if activate is None:
                        logger.critical("Activation value of model %s for backend %s must be specified.", name, backend)
                    tmp_config[backend][name]["activate"] = activate

        asserted_config = {}

        # Now compare user and default configuration
        for backend, impl in tmp_config.items():
            # If user didn't specify anything for this backend continue and use defaults
            if backend not in asserted_config:
                asserted_config[backend] = {}
            for model, parameters in impl.items():
                # If user didn't specify details of a model continue and use defaults
                # Check default activation status
                if not parameters["activate"]:
                    continue
                if model not in asserted_config[backend]:
                    asserted_config[backend][model] = parameters["default"]
                # Just skip and use defaults if user hasn't specified anything
                if backend not in user_config or model not in user_config[backend]:
                    continue
                # After stripping the activation flag the parameter list in user_config the length
                # of the parameter list must be the same.
                if len(user_config[backend][model]) != len(asserted_config[backend][model]):
                    logger.critical("Parameter list for %s model %s differs", backend, model)
                # Replace defaults with user settings if any
                for u in asserted_config[backend][model]:
                    asserted_config[backend][model][u] = user_config[backend][model].get(u, asserted_config[backend][model][u])


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
            print_dict(self.run_config)
        if model_config_flag:
            print_dict(self.model_config)
        print("====================================================")
