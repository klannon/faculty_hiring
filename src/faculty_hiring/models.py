# -*- coding: utf-8 -*-
"""
This file contains various models used for studying faculty hiring.
"""
import logging, sys
from collections import namedtuple, Counter
import math, statistics
import numpy as np
from faculty_hiring import strategy

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
        self.attributes = {'quality':None, 'mask_':None}
        self.attrib_dtypes = {'quality':np.dtype(float), 'mask_':np.dtype(bool)}
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
            elif attrib == 'mask_':
                attrib_vals.append(True)
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

class Faculty:
    """Data structure to represent a member of the faculty"""
    def __init__(self, age=None, exp=0, **properties):

        self._fields = ['age','experience']
        
        if age != None:
            self.age = age
        else:
            # A randomly generated new assistant professor
            self.age = int(np.random.normal(33,1))

        # Used for things like deciding when a faculty member needs to face a tenure decision
        self.experience = exp
        
        for prop in properties:
            self._fields.append(prop)
            setattr(self,prop,properties[prop])

    def ready_to_retire(self):
        """Indicates whether this faculty member is ready to retire."""

        if self.age < 65:
            return False

        if self.age > 85:
            return True
        
        prob = min(0.98,0.05 + 0.02*(self.age-65))
        if np.random.uniform() > prob:
            return False
        else:
            return True

    def advance_age(self,years=1):
        """Advance the faculty age by the indicated number of years"""
        self.age += years
        self.experience += years

class Department:
    """Data structure to represent all the faculty"""

    def __init__(self, population, lines, unfilled_lines=None, threshold=3.):
        """Constructor

        Args:
           lines (Counter): The number of lines in each field
           population (CandidatePopulation): The distribution of candidates 
                from which to fill out the faculty population.
           threshold (float): The mininum quality for a faculty member
        """

        if 'field' not in population.record_type._fields:
            raise AttributeError('field missing from population attributes')
        
        self.lines = Counter(lines)
        if unfilled_lines == None:
            self.unfilled_lines = Counter()
            unfilled_lines = Counter()
        else:
            self.unfilled_lines = Counter(unfilled_lines)
        
        # We're going to fill up the faculty by running searches until all the lines are filled
        # Start by making a list of positions to search for.  It's like a predetermined list of retirements
        self.faculty = []
        lines_to_fill = list((self.lines - self.unfilled_lines).elements())
        # Calculate the average number of searches we'll need to run
        # to fill the department up in about 25 years
        search_rate = len(lines_to_fill)/30.
        np.random.shuffle(lines_to_fill)
        year = 0
        while len(lines_to_fill) > 0:
            _logger.debug('{:3d}: Dept size: {:2d}, '.format(year,len(self.faculty)) + 
                          'Lines to fill: {:2d}, '.format(len(lines_to_fill))+
                          'Unfilled Lines: {:2d}'.format(sum(self.unfilled_lines.values())))
            year+=1
            #Start an annual hiring cycle.  Let's generate the population!
            # Assumption:  There are about 300 people applying for our openings.
            pop = population.generate_population(300)
            
            n_searches = min(np.random.poisson(search_rate),len(lines_to_fill))
            failed_searches = []
            for i in range(n_searches):

                # Get the field in which we're searching for this search
                search_field = lines_to_fill.pop()

                # Run the search
                new_hire  = strategy.pick_best(population.record_type, pop, {'field':search_field},
                                               threshold=threshold)
                if new_hire != None:
                    self.faculty.append(Faculty(**new_hire._asdict()))
                else:
                    # Failed hire, need to repeat the search
                    failed_searches.append(search_field)

            lines_to_fill.extend(failed_searches)
            self.next_year()

            #Replace the retirements
            retirements = self.unfilled_lines - unfilled_lines
            self.unfilled_lines = Counter(unfilled_lines)
            lines_to_fill.extend(retirements.elements())
            
            
    def next_year(self):
        """Move the department forward by one year.  Everyone gets one year older.  
           Some faculty may retire.
        """
        # Age the faculty
        retirements = []
        for f in self.faculty:
            f.advance_age()
            if f.ready_to_retire():
                retirements.append(f)

        for f in retirements:
            self.unfilled_lines[f.field]+=1
            self.faculty.remove(f)

    def summary(self):
        """Just prints out some info about this department"""
        print('This department has '+
              '{} faculty and {} unfilled lines.'.format(len(self.faculty),
                                                         sum(self.unfilled_lines.values())))
        
        if (len(self.faculty)>0):
            print('Current Faculty:')
            print('----------------')

            print_fields = [f for f in self.faculty[0]._fields if not f.endswith('_')]
            hdr_list = []
            fmt_list = []
            divider_list = []
            for f in print_fields:
                # Check the size of the field name
                l = len(f)
                # Now check the length of the field values
                if isinstance(getattr(self.faculty[0],f),str):
                    lmax = max([len(getattr(fac,f)) for fac in self.faculty])
                    l = max(l,lmax)
                elif isinstance(getattr(self.faculty[0],f),int):
                    lmax = max([int(1+math.log10(getattr(fac,f)+1e-10)) for fac in self.faculty])
                    l = max(l,lmax)
                elif isinstance(getattr(self.faculty[0],f),float):
                    lmax = max([int(1+math.log10(getattr(fac,f)+1e-10)) for fac in self.faculty])
                    lmax += 3
                    l = max(l,lmax)
                divider_list.append(l*'-')
                hdr = '{:'
                hdr += str(l)
                hdr += '}'
                hdr_list.append(hdr)
                fmt = '{:'
                fmt += str(l)
                if isinstance(getattr(self.faculty[0],f),float):
                    fmt += '.2f'
                fmt += '}'
                fmt_list.append(fmt)

            hdr_str = '  '.join(hdr_list)
            print(hdr_str.format(*print_fields))
            print('  '.join(divider_list))
            fmt_str = '  '.join(fmt_list)
            for f in self.faculty:
                vals = [getattr(f,field) for field in print_fields]
                print(fmt_str.format(*vals))

            print('Summary Statistics:')
            maxlen = max([len(f) for f in print_fields])
            for f in print_fields:
                # Only know how to handle these cases
                if isinstance(getattr(self.faculty[0],f),str):
                    counts = Counter([getattr(fac,f) for fac in self.faculty])
                    stats = ['{}-{:4.1f}%'.format(v, 100*counts[v]/sum(counts.values()))
                             for v in sorted(counts.keys())]
                    print(('  {:'+str(maxlen)+'}: {}').format(f,', '.join(stats))) 
                elif isinstance(getattr(self.faculty[0],f),(int,float)):
                    ave_v = statistics.mean([getattr(fac,f) for fac in self.faculty])
                    min_v = min([getattr(fac,f) for fac in self.faculty])
                    max_v = max([getattr(fac,f) for fac in self.faculty])
                    print(('  {:'+str(maxlen)+'}:').format(f) +
                          ' Ave {:.2f}, Min {:.2f}, Max {:.2f}'.format(ave_v, min_v, max_v))
                    
            print('')
            print('Unfilled Lines:')
            for f in self.unfilled_lines.keys():
                if self.unfilled_lines[f] > 0:
                    print('{}: {:3d}'.format(f,self.unfilled_lines[f]))

                      
