import numpy as np

def pick_best(record_type, population, filter_criteria=None,
              rank_attrib='quality', threshold = 0.0, remove=True):
    """Take the best candidate, optionally filtering on some critera

    Args:
      record_type (type): The namedtuple type for filling in the returned cadidate.
      population (namedtuple): The list of candidates to pick from
      filter_criteria (dict): A dictionary specifying values for attributes 
          to filter the candidates on.
      rank_attrib (str): The attribute to use for sorting candidates.
      threshold (float): The minimum value to require for the selected
          candidate.  If no candidate exceeds this threshold, then
      remove (bool): Whether to mark the candidate as gone once picked
          return None.
    """

    rank = getattr(population,rank_attrib)

    # This gets the list of indices sorted in descending order
    # according to the rank variable
    order = np.argsort(rank)[::-1]

    # This selects for us the parts of the population above threshold
    # and matching the desired criteria
    mask = (population.mask_ & (rank > threshold))
    if filter_criteria != None:
        for key, value in filter_criteria.items():
            if isinstance(value,(tuple,list)):
                tmp_mask = np.zeros_like(population.mask_,dtype=bool)
                for x in value:
                    tmp_mask |= (getattr(population,key)==x)
                mask &= tmp_mask
            else:
                mask &= (getattr(population,key)==value)
    if np.count_nonzero(mask) == 0:
        return None
            
    # This picks out the index of the top candidate after filtering
    # Note, the "mask[order]" is required to rearrange the mask so
    # that it corresponds to the sorted order of the order array.
    best_ind = order[mask[order]][0]

    # Now, let's return the record for the best candidate
    vals = []
    for f in record_type._fields:
        if f == 'mask_':
            vals.append(best_ind)
        else:
            vals.append(getattr(population,f)[best_ind])
    if remove:
        population.mask_[best_ind] = False
        
    return record_type(*vals)
    
def pick_pref(record_type, population, preference_criteria, rank_tolerance,
              filter_criteria=None,
              rank_attrib='quality', threshold = 0.0, remove=True):
    """Take the preferred candidate if not worse than "rank_tolerance" from 
    the best candidate.

    Args:
      record_type (type): The namedtuple type for filling in the returned cadidate.
      population (namedtuple): The list of candidates to pick from
      preference_criteria (dict): Lists the attributes of the preferred candidate.
      rank_tolerance (float): The preferred candidate will only be taken over the 
          best only if within this amount of the best.  Otherwise best is chosen.
      filter_criteria (dict): A dictionary specifying values for attributes 
          to filter the candidates on.
      rank_attrib (str): The attribute to use for sorting candidates.
      threshold (float): The minimum value to require for the selected
          candidate.  If no candidate exceeds this threshold, then
          return None.
    """

    # Some sanity checking
    if isinstance(rank_tolerance,(tuple,list)):
        if isinstance(preference_criteria, (tuple,list)):
            if len(rank_tolerance) != len(preference_criteria):
                raise IndexError('preference_criteria and rank_tolerance '+
                                 'both have to have compatible lengths')
        else:
            raise IndexError('preference_criteria is a scalar while rank_tolerance is a vector.')
    
    best = pick_best(record_type, population,filter_criteria, rank_attrib, threshold, remove=False)

    # If the overall best didn't pass muster, then no one did
    if best == None:
        return None

    if not isinstance(preference_criteria,(tuple, list)):
        preference_criteria = (preference_criteria,)
    if not isinstance(rank_tolerance,(tuple,list)):
        rank_tolerance = len(perference_criteria)*[rank_tolerance]

    # We check each one in order.  The first one that satisfies is returned.
    for pc, tol in zip(preference_criteria,rank_tolerance):
    
        if filter_criteria == None:
            pref_filter = pc
        else:
            # Merge the two.  This requires some care because if the
            # filter and the preference dictionaries both specify a
            # property they either have to be rectified.  If they
            # can't be rectified, then this preference can't be
            # satisfied
            pref_filter = dict(filter_criteria)
            for k in pc.keys():
                if not k in pref_filter:
                    pref_filter[k] = pc[k]
                else:
                    fv = pref_filter[k]
                    pv = pc[k]
                    if isinstance(fv,(tuple,list)):
                        if isinstance(pv,(tuple,list)):
                            intersect = []
                            for v in pv:
                                if v in fv:
                                    intersect.append(v)
                            if len(intersect) == 0:
                                # There is no overlap, so no candidate can satisfy these requirements
                                pref_filter = None
                            pref_filter[k] = intersect
                        else:
                            if pv in fv:
                                pref_filter[k] = pv
                            else:
                                # No overlap
                                pref_filter = None
                    else:
                        if isinstance(pv,(tuple,list)):
                            if fv in pv:
                                pref_filter[k] = fv
                            else:
                                # No overlap
                                pref_filter = None
                        else:
                            if fv == pv:
                                pref_filter[k] = fv
                            else:
                                pref_filter = None
        if pref_filter == None:
            # The differences between our requirements and our
            # preferences couldn't be mutually satisifed so we skip
            # this search
            continue
                                
        pref = pick_best(record_type, population, pref_filter, rank_attrib, threshold, remove=False)

        if pref != None:    
            if pref.quality + tol > best.quality:
                if remove:
                    # The mask_ field in an individual record give the index in the popluation.
                    population.mask_[pref.mask_] = False
                return pref

    else:
        # If I'm here, none of our preferences were selected, so return the overall best!
        if remove:
            # The mask_ field in an individual record give the index in the popluation.
            population.mask_[best.mask_] = False
        return best
