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

    best = pick_best(record_type, population,filter_criteria, rank_attrib, threshold, remove=False)

    if filter_criteria == None:
        pref_filter = preference_criteria
    else:
        # Merge the two, taking preference over filter where the two contradict
        pref_filter = {**filter_criteria, **preference_criteria}
        
    pref = pick_best(record_type, population, pref_filter, rank_attrib, threshold, remove=False)

    if pref == None:
        if best != None and remove:
            # The mask_ field in an individual record give the index in the popluation.
            population.mask_[best.mask_] = False
        return best

    # If the overall best didn't pass muster, then no one did
    if best == None:
        return None
    
    if pref.quality + rank_tolerance > best.quality:
        # OK, this is dumb.  I don't have an easier way to remove the
        # selected candidate than to re-run the selection...
        if remove:
            # The mask_ field in an individual record give the index in the popluation.
            population.mask_[pref.mask_] = False
        return pref
    else:
        # OK, this is dumb.  I don't have an easier way to remove the
        # selected candidate than to re-run the selection...
        if remove:
            # The mask_ field in an individual record give the index in the popluation.
            population.mask_[best.mask_] = False
        return best
