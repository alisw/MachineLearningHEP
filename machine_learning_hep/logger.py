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
Methods to: provide and manage central logging utility
"""
import logging
import sys
from copy import copy


class ExitHandler(logging.Handler):
    """
    Add custom logging handler to exit on certain logging level
    """
    def emit(self, record):
        logging.shutdown()
        sys.exit(1)

class MLLoggerFormatter(logging.Formatter):
    """
    A custom formatter that colors the levelname on request
    """
    # color names to indices
    color_map = {
        'black': 0,
        'red': 1,
        'green': 2,
        'yellow': 3,
        'blue': 4,
        'magenta': 5,
        'cyan': 6,
        'white': 7,
    }

    level_map = {
        logging.DEBUG: (None, 'blue', False),
        logging.INFO: (None, 'green', False),
        logging.WARNING: (None, 'yellow', False),
        logging.ERROR: (None, 'red', False),
        logging.CRITICAL: ('red', 'white', True),
    }
    csi = '\x1b['
    reset = '\x1b[0m'

    # Define default format string
    def __init__(self, fmt='%(levelname)s in %(pathname)s:%(lineno)d:\n ↳ %(message)s',
                 datefmt=None, style='%', color=False):
        logging.Formatter.__init__(self, fmt, datefmt, style)
        self.color = color

    def format(self, record):
        # Copy the record so the global format is kept
        cached_record = copy(record)
        requ_color = self.color
        # Could be a lambda so check for callable property
        if callable(self.color):
            requ_color = self.color()
        # Make sure levelname takes same space for all cases
        cached_record.levelname = f"{cached_record.levelname:8}"
        # Colorize if requested
        if record.levelno in self.level_map and requ_color:
            bg, fg, bold = self.level_map[record.levelno]
            params = []
            if bg in self.color_map:
                params.append(str(self.color_map[bg] + 40))
            if fg in self.color_map:
                params.append(str(self.color_map[fg] + 30))
            if bold:
                params.append('1')
            if params:
                cached_record.levelname = "".join((self.csi, ';'.join(params), "m",
                                                   cached_record.levelname,
                                                   self.reset))
        return logging.Formatter.format(self, cached_record)


def configure_logger(debug, logfile=None):
    """
    Basic configuration adding a custom formatted StreamHandler and turning on
    debug info if requested.
    """
    logger = logging.getLogger("MachinelearningHEP")
    if logger.hasHandlers():
        return

    # Turn on debug info only on request
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    formatter = MLLoggerFormatter(color=lambda : getattr(sh.stream, 'isatty', None)) # pylint: disable=C0326

    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Add logfile on request
    if logfile is not None:
        # Specify output format
        fh = logging.FileHandler(logfile)
        fh.setFormatter(MLLoggerFormatter())
        logger.addHandler(fh)

    # Add handler to exit at critical. Do this as the last step so all former
    # logger flush before aborting
    logger.addHandler(ExitHandler(logging.CRITICAL))


def get_logger():
    """
    Get the global logger for this package and set handler together with formatters.
    """
    configure_logger(False, None)
    return logging.getLogger("MachinelearningHEP")
