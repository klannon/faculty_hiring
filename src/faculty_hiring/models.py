# -*- coding: utf-8 -*-
"""
This file contains various models used for studying faculty hiring.
"""
import logging
from collections import namedtuple
import numpy as np

from faculty_hiring import __version__

__author__ = "Kevin Lannon"
__copyright__ = "Kevin Lannon"
__license__ = "mit"

_logger = logging.getLogger(__name__)

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


class CandidatePopulation:
    """Provides a model for the population of potential faculty in which we are searching."""

    def __init__(self):
        self.attributes = {'quality':None}
        self.attrib_dtypes = {'quality':np.dtype(float)}
        self.record_type = None

    def add_attribute(self,name,values):
        """Add an attribute to this population

        Args:
          name (str): The attribute name (e.g. gender)
          values (dict): A dictionary mapping the values to probabilities
                         E.g. {"male":0.5, "female":0.5}
        """

        # Don't allow attempts to add multiple attributes of the same type.
        if name in self.attributes:
            msg = 'Attribue {} already defined.'.format(name)
            raise ValueError(msg)

        # Figure out what type to use for attribute data
        t = type(sorted(values.keys())[0])
        if not all([isinstance(x,t) for x in values.keys()]):
            msg = 'Attribute {} has inconstistent value types.\n'.format(name)
            msg += 'Values = {}'.format(values)
            raise TypeError(msg)

        # Decode the type
        if t == int:
            self.attrib_dtypes[name] = np.dtype(int)
        elif t == float:
            self.attrib_dtypes[name] = np.dtype(float)
        elif t == str:
            maxlen = max([len(x) for x in values.keys()])
            type_str = 'U{:d}'.format(maxlen)
            self.attrib_dtypes[name] = np.dtype(type_str)
        else:
            msg =  "Attribute {} has ".format(name)
            msg += "values of type {} ".format(t)
            msg += "which isn't currently handled."
            raise TypeError(msg)

        # Store the attribute values
        self.attributes[name] = values
        self.record_type = namedtuple('Candidate',
                                      sorted(self.attributes.keys()))


    def generate_candidate(self, min_quality = 2.0):
        """Randomly generate the attributes of one faculty candidate"""

        # Generate the attribute values and store them on a list
        attrib_vals = []
        for attrib in sorted(self.attributes.keys()):
            if attrib == 'quality':
                # Generate the candidate quality score.
                # We model "quality" with a unit normal distribution.
                # But we assume faculty candidates are already out on
                # the 3 sigma tail.
                # I know this is computationally wasteful, but it's easy
                quality = abs(np.random.standard_normal())
                while quality < min_quality:
                    quality = abs(np.random.standard_normal())
                attrib_vals.append(quality)
            else:
                values = self.attributes[attrib]
                val = np.random.choice(list(values.keys()),p=list(values.values()))
                attrib_vals.append(val)

        return self.record_type(*attrib_vals)

    def generate_population(self,num, fluctuate = True):
        """Create a population of "num" candidates.

        Args:
           num: Number of candidates to generate.
        """

        if fluctuate:
            num = np.random.poisson(num)
        
        # Create numpy arrays to hold attributes of the candidates
        # in this population (i.e. the columns)
        attrib_vals = []
        for attrib in sorted(self.attributes.keys()):
            dt = self.attrib_dtypes[attrib]
            attrib_vals.append(np.empty(num,dt))

        p = self.record_type(*attrib_vals)

        for i in range(num):
            r = self.generate_candidate()
            for attrib in sorted(self.attributes.keys()):
                getattr(p,attrib)[i] = getattr(r,attrib)

        return p

