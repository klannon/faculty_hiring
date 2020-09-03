# -*- coding: utf-8 -*-
"""
This file contains various models used for studying faculty hiring.
"""
import logging, sys
from collections import namedtuple, Counter
import math, statistics
import numpy as np
from scipy import special
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

        # Store the attribute values
        self.attributes[name] = values
        self.record_type = namedtuple('Candidate',
                                      sorted(self.attributes.keys()))

    def generate_candidate(self, min_quality = 2.0):
        """Randomly generate the attributes of one faculty candidate

        This used to be a building block method, but not I'm just
        leaving it here for posterity as all the work is done in
        `gerenate_population()`.
        """

        # Use generate population to make a size=1 population
        x = self.generate_population(1, fluctuate=False, min_quality=min_quality)

        # Since we're expecting a record with simple types rather than
        # length=1 Numpy arrays, just do a little shenanegins to get
        # this sorted out.
        attrib_vals = []
        for attrib in sorted(self.attributes.keys()):
            attrib_vals.append(getattr(x,attrib)[0])

        return self.record_type(*attrib_vals)

    def generate_population(self,num, fluctuate = True, min_quality=2.0):

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
            if attrib == 'quality':
                # This voodoo generates a random number on a normal
                # distribution, only for the positive tail above
                # "min_quality"
                sq2 = math.sqrt(2)
                l = special.erf(min_quality/sq2)
                x = np.random.uniform(low=l, high=1.0, size=num)
                attrib_vals.append(sq2*special.erfinv(x))
            elif attrib == 'mask_':
                attrib_vals.append(np.full(num, True))
            else:
                values = self.attributes[attrib]
                val = np.random.choice(list(values.keys()),p=list(values.values()),size=num)
                attrib_vals.append(val)

        p = self.record_type(*attrib_vals)

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

    def __init__(self, population, lines, unfilled_lines=None, threshold=3.,
                 minimum_open_lines = 0):
        """Constructor

        Args:
           lines (Counter): The number of lines in each field
           population (CandidatePopulation): The distribution of candidates 
                from which to fill out the faculty population.
           threshold (float): The mininum quality for a faculty member
           minimum_open_lines (int): The smallest number of open lines a field can have.  
                Set to negative to allow fields to borrow lines from others.
        """

        if 'field' not in population.record_type._fields:
            raise AttributeError('field missing from population attributes')
        
        self.lines = Counter(lines)
        if unfilled_lines == None:
            self.unfilled_lines = Counter()
            unfilled_lines = Counter()
        else:
            self.unfilled_lines = Counter(unfilled_lines)

        self.open_min = minimum_open_lines
            
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
            
            
    def add_faculty(self, candidate, new_line=False, convert_line=None):
        """Add a new member to the faculty.  If there is an unfilled line in 
        the candidate's field, decrement that unfilled line count.  Otherwise
        either increase the number of total lines or convert a line, as directed
        by the arguments.  Specifying `new_line` or `convert_line` will do nothing 
        in the case that an appropriate unfilled line is available.

        Args:
           candidate (namedtuple): The demographic and quality information for the new hire.
           new_line (bool): If true, will increase the faculty lines if none are open.
           convert_line (str): If not `None` will swap an unused line from the specified field.
               Can also specify a tuple of strings to allow swapping from a list of fields, 
               in the order specified in the tuple        
        """
        self.faculty.append(Faculty(**candidate._asdict()))
        if self.unfilled_lines[candidate.field] > self.open_min:
            self.unfilled_lines[candidate.field]-=1
        else:
            # Hey, looks like we've hired someone into a line that didn't exist
            self.lines[candidate.field]+=1
            if not new_line:

                if sum(self.unfilled_lines.values()) == 0:
                    raise ValueError('No available line for this new hire.')
                
                if isinstance(convert_line, (tuple,list)):
                    for f in convert_line:
                        if self.unfilled_lines[f] > self.open_min:
                            break
                    else:
                        # Pick a random field
                        f = np.random.choice(list(self.unfilled_lines.elements()))

                elif isinstance(convert_line, str):
                    if self.unfilled_lines[convert_line] > self.open_min:
                        f = convert_line
                    else:
                        # Pick a random field
                        f = np.random.choice(list(self.unfilled_lines.elements()))
                else:
                    # Pick a random field
                    f = np.random.choice(list(self.unfilled_lines.elements()))

                # Reduce the unfilled line in the chosen field
                self.unfilled_lines[f]-= 1
                self.lines[f]-= 1
                    
                                             
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
                print('{}: {:3d}'.format(f,self.unfilled_lines[f]))
                    
