# -*- coding: utf-8 -*-
"""
This file contains implementations of various election approaches
"""

__author__ = "Kevin Lannon"
__copyright__ = "Kevin Lannon"
__license__ = "mit"

import numpy as np

def choose_fptp(votes, num_to_choose=1):
    """Simple 'First past the post' voting:
    https://en.wikipedia.org/wiki/First-past-the-post_voting

    Args:
        votes(numpy.array): Array of votes.  First index is the voter 
           number and second index is for choices.  Any number > 0 and 
           <= num_to_choose counts as a vote for that candidate.  (This 
           allows us to interpret a ranked list in FPTP voting.
    
        num_to_choose(int): Number of candidates to choose.
    """

    totals = np.count_nonzero((votes > 0) & (votes <= num_to_choose), axis=0)

    ind = -num_to_choose
    rank = totals.argsort()
    
    # Check for a tie with the last place choice, and report all of
    # the tied options as well.
    while (ind > -len(totals)) and (totals[rank[ind]] == totals[rank[ind-1]]):
        ind -= 1

    # This will return them in order of highest vote getters
    return rank[:ind-1:-1]

def choose_irv(votes):
    """Instant Run-off voting:
    https://en.wikipedia.org/wiki/Instant-runoff_voting

    Args:
        votes(numpy.array): Array of votes.  First index is the voter 
           number and second index is for choices.  
    """

    totals = np.sum(votes == 1, axis=0)
    max_votes = np.max(totals)
    v_scratch = votes.copy()
    n_cands = votes.shape[1]
    remaining = np.ones_like(totals,dtype=bool)
    
    while max_votes <= 0.5 * np.sum(totals) and np.sum(remaining) > 1:

        min_votes = np.min(totals[remaining])
        to_remove = np.argwhere(totals == min_votes).flatten()

        # Check that we're not about to remove the last remaining ones
        if len(to_remove) == np.sum(remaining):
            return to_remove

        remaining[to_remove] = False

        for ind in to_remove:
        
            # Remove the votes for the candidate with the lowest total (or
            # tied with the lowest) and slide all other votes one rank
            # better
            offset = np.where(((v_scratch > 1) &
                               (v_scratch > v_scratch[:,ind][:,np.newaxis])),1,0)
            offset[:,ind] = v_scratch[:,ind]
            v_scratch = v_scratch - offset

        # Recount the votes
        totals = np.sum(votes == 1, axis=0)
        max_votes = np.max(totals)

    else:
        # Either we've got someone with a majority or we've eliminated
        # too many people
        max_votes = np.max(totals[remaining])
        winners = np.argwhere(totals == max_votes).flatten()
        return winners


def choose_condorcet(votes):
    """Basic implementation of Condorcet voting
    https://en.wikipedia.org/wiki/Condorcet_method

    Args:
        votes(numpy.array): Array of votes.  First index is the voter 
           number and second index is for choices.  

    Returns:
        Array of winner indices (more than one if a tie)
    
    """

    # Ranking a candidate 0 means not voting.  Make sure that rank always loses
    v = np.where(votes<=0,votes.shape[1]+100,votes)
    
    # Construct the pairwise compairsion matrix and sum it
    comp = np.sum(v[:,np.newaxis,:] > v[:,:,np.newaxis],axis=0)

    # Now how many head to head victories each candidate had
    victs = np.sum((comp - comp.T) > 0, axis=1)
    rank = victs.argsort()

    # The winner is the one with the most head-to-head victories
    # Check for ties in the number of head to head victories and
    # return all ties.
    tie_ind = -1
    while tie_ind > -len(rank)+1 and victs[rank[-1]] == victs[rank[tie_ind-1]]:
        tie_ind -= 1

    return rank[tie_ind:]

def choose_schulze(votes, num_to_choose=1):
    """Schulze voting method, which is a varient on Condorcet voting that 
    avoids cycles.
    https://en.wikipedia.org/wiki/Schulze_method

    Impelemtation adapted from here: 
        https://en.wikipedia.org/wiki/Schulze_method#Implementation

    Args:
        votes(numpy.array): Array of votes.  First index is the voter 
           number and second index is for choices.
        num_to_choose(int): Indicates the number of seats to fill

    Returns:
        Array of winner indices
    
    """

    # Ranking a candidate 0 means not voting.  Make sure that rank always loses
    v = np.where(votes<=0,votes.shape[1]+100,votes)
    
    # Construct the pairwise compairsion matrix and sum it
    comp = np.sum(v[:,np.newaxis,:] > v[:,:,np.newaxis],axis=0)

    # Next, let's compute the path stength first by considering those
    # with direct connections
    ps = np.where(comp > comp.T,comp,0)

    # Now we need to check the 2-hop path stengths I couldn't find a
    # numpy way to do this.  It should be a pretty small array.
    # Possibly this is a candidate for numba?
    for i in range(ps.shape[0]):
        for j in range(ps.shape[0]):
            if i != j:
                for k in range(ps.shape[0]):
                    if i != k and j != k:
                        ps[j,k] = max(ps[j,k],min(ps[j,i],ps[i,k]))

    # Now figure out which candidates are better than which
    wins = np.sum(ps > ps.T,axis=1)
    rank = wins.argsort()
    
    ind = -num_to_choose
    # Check for a tie with the last place choice, and report all of
    # the tied options as well.
    while (ind > -len(wins)) and (wins[rank[ind]] == wins[rank[ind-1]]):
        ind -= 1

    return rank[:ind-1:-1]

def choose_borda(votes, num_to_choose=1, start_from_zero=False):
    """Borda voting is a point-based voting scheme
    https://en.wikipedia.org/wiki/Borda_count

    Args:
        votes(numpy.array): Array of votes.  First index is the voter 
           number and second index is for choices.
        num_to_choose(int): Indicates the number of seats to fill
        start_from_zero(bool): Whether the lowest rank gets 1 point or 0

    Returns:
        Array of winner indices
    
    """

    n_cands = votes.shape[1]
    offset = 0 if start_from_zero else 1
    scores = np.where((votes > 0) & (votes <= n_cands),n_cands+offset-votes,0)
    totals = np.sum(scores,axis=0)
    rank = totals.argsort()

    # Check for a tie with the last place choice, and report all of
    # the tied options as well.
    ind = -num_to_choose
    while (ind > -len(totals)) and (totals[rank[ind]] == totals[rank[ind-1]]):
        ind -= 1

    # This will return them in order of highest vote getters
    return rank[:ind-1:-1]
    
def choose_score_voting(scores, num_to_choose=1):
    """Score voting is a cardinal voting scheme that awards the election 
    to the candidate with the highest point total, where candidates can be
    scored independently of one another.

    https://en.wikipedia.org/wiki/Score_voting

    Args:
        scores(numpy.array): Array of scores.  First index is the voter 
           number and second index is for choices.
        num_to_choose(int): Indicates the number of seats to fill

    Returns:
        Array of winner indices
    
    """

    totals = np.sum(scores,axis=0)
    rank = totals.argsort()

    # Check for a tie with the last place choice, and report all of
    # the tied options as well.
    ind = -num_to_choose
    while (ind > -len(totals)) and (totals[rank[ind]] == totals[rank[ind-1]]):
        ind -= 1

    # This will return them in order of highest vote getters
    return rank[:ind-1:-1]
    
def choose_majority_judgment(scores):
    """Majority judgment is a cardinal voting scheme very similar to 
    score voting except that median score instead of total score is used
    to decide the election.  If multiple voters are tied, you remove 
    people with score = median score until you have one unique top 
    candidate.

    https://en.wikipedia.org/wiki/Majority_judgment

    Args:
        scores(numpy.array): Array of scores.  First index is the voter 
           number and second index is for choices.

    Returns:
        The winner in a one element array, just to be like other voting 
        functions.
    
    """

    # We're going to use a masked array to run this election because
    # we need to remove voters from the lists until we find a winner.
    s = np.ma.array(scores,mask=False)
    
    # Start by getting the medians and find which have the maximum value
    medians = np.ma.median(s,axis=0)
    best_med = np.max(medians)
    best_ind = np.argwhere(medians == best_med).flatten()
    
    # Now progressively remove entries until only one candidate
    # has the best median
    while s[:,best_ind].count() > 0 and len(best_ind) > 1:
        # Let's mask off the non-contenders
        s[:,np.argwhere(medians != int(best_med)).flatten()] = np.ma.masked
        
        # Now we need to mask off voters who scored the medians until
        # one of the medians changes
        while s[:,best_ind].count() > 0 and np.all(medians[best_ind]==best_med):
            for ind in best_ind:
                # Find an entry to remove
                meds = np.argwhere(s[:,ind]==int(best_med)).flatten()
                if len(meds)>0:
                    s[meds[0],ind] = np.ma.masked
                else:
                    # Didn't find an item equal to the median.  Remove
                    # the next bigger item.
                    val = np.min(s[:,ind][s[:,ind]>best_med])
                    i = np.argwhere(s[:,ind]==val).flatten()[0]
                    s[i,ind] = np.ma.masked
            medians = np.ma.median(s,axis=0)
        best_med = np.max(medians)
        best_ind = np.argwhere(medians == best_med).flatten()
        # Check to make sure that we haven't dropped to no "bests."  If we have,
        # return whatever's left
        return np.flatnonzero(~medians.mask)
    else:
        return best_ind

def choose_star(scores):
    """Score then Automatic Runoff (STAR) voting is another cardinal voting 
    approach that differs from others in that the top two in terms of 
    scores go into an "automatic runoff" where "votes" are tallied based on
    which candidate got a higher score voter-by-voter.

    https://en.wikipedia.org/wiki/STAR_voting

    Args:
        scores(numpy.array): Array of scores.  First index is the voter 
           number and second index is for choices.

    Returns:
        The winner in a one element array, just to be like other voting 
        functions.
    
    """

    # Start with a simple score-based election, already implemented
    finalists = choose_score_voting(scores,2)

    # OK, for the automated run-off, I'm making a choice because I don't
    # see any details about this.  In counting votes, if a voter scores
    # two candidates the same, I'm giving them both votes.  The alternative
    # (give neither one a vote) is harder to implement.  In a two-person
    # race, either choice is equivalent.  If we have 3 or more (because
    # of a tie in the score voting stage) then I think giving tied
    # candidates both votes is better than leaving them off (e.g. if 99% of
    # people score A and B at 10 and C at 1, but 1% score A, B, and C at 1, 2,
    # and 10 respectively, I don't think C should win.)
    max_scores = np.max(scores[:,finalists],axis=1)
    votes = (scores[:,finalists] == max_scores.reshape(-1,1))
    totals = np.sum(votes,axis=0)
    rank = totals.argsort()

    # Check for a tie with the last place choice, and report all of
    # the tied options as well.
    ind = -1
    while (ind > -len(totals)) and (totals[rank[ind]] == totals[rank[ind-1]]):
        ind -= 1

    # This will return them in order of highest vote getters
    return finalists[rank[:ind-1:-1]]
    
def choose_stv(votes, num_to_choose=1):
    """Single Transferrable Vote method, which can be used to elect multiple
    candidates from a pool.  If only one candidate is chosen, it's basically
    identical to IRV.

    https://en.wikipedia.org/wiki/Single_transferable_vote

    There are a lot of moving parts here, and too many options to try to 
    implement them all.  I will therefore be making some arbitrary choices, 
    but when I do, I'll try to stay close to the Fair Vote description.

    https://www.fairvote.org/multi_winner_rcv_example

    First, we need to decide on a quota.  Based on reading too much on 
    Wikipedia, I've concluded that FairVote is actually using the 
    Hagenbach-Bischoff quota (even though one could also call it the 
    Droop quota).  Anyway, this is a computer program and I can do 
    fractional votes, so we're going with Hagenbach-Bischoff.

    https://en.wikipedia.org/wiki/Hagenbach-Bischoff_quota

    Transferring of votes will be done by fractionally weighting the next 
    preference of all votes according to the Gregory method because this is 
    what most closely matches the description at Fair Vote.

    https://en.wikipedia.org/wiki/Counting_single_transferable_votes#Gregory

    Because I can't think of a sane way to do it otherwise, when a
    candidate is elected or eliminated, they'll be removed from the
    ballot and can't have votes subsequently transferred to them.
    Also, when calculating the fraction, I won't be correcting for
    ballots with no next preference listed, as seems to be suggested
    in some versions of the Gregory method described in Wikipedia.  

    Args:
        votes(numpy.array): Array of votes.  First index is the voter 
           number and second index is for choices.  
    
        num_to_choose(int): Number of candidates to choose.

    """

    # Start by making a copy of the votes because in the counting we're going
    # to modify the array
    v = votes.copy()
    
    #  we need an array to store the weights
    weights = np.ones((v.shape[0],1),dtype=float)

    # Keep the winners in their own array
    winners = []

    # Determine the quota
    quota = v.shape[0]/(num_to_choose + 1)

    # Store which candidates are still in the running
    remaining = np.ones(v.shape[1],dtype=bool)

    # Now run rounds of removal of winners and/or losers
    while len(winners) < num_to_choose:

        # If the remaining candidates equals the number of seats left
        # to fill, we can be done.
        if num_to_choose - len(winners) == sum(remaining):
            winners.extend(np.argwhere(remaining).flatten())
            break

        # Tabulate the votes
        totals = np.sum(np.where(v==1,weights,0),axis=0)
        
        # Find new winners
        new_winners = np.argwhere(totals > quota).flatten()
        winners.extend(new_winners)
        remaining[new_winners] = False
        
        # Determine and update weights, but need to do it in order of most votes
        sorted_winners = np.argsort(totals[new_winners])
        for ind in reversed(new_winners[sorted_winners]):

            # We need a weighting factor for this rule
            w_new = np.where(v[:,ind]==1, (totals[ind]-quota)/totals[ind], 1).reshape(-1,1)
            weights = weights*w_new

            # Now update the vote ranks.
            offset = np.where(((v > 1) & (v > v[:,ind][:,np.newaxis])),1,0)
            offset[:,ind] = v[:,ind]
            v = v - offset
            
        # If there were no winners, remove the lowest totals
        if len(new_winners) == 0:

            # Find the lowest of the remaining
            losers = np.argwhere(totals == np.min(totals[remaining])).flatten()
            
            # Handle a special case where an N-way tie for last would
            # prevent us from filling all the seats
            if np.sum(remaining) - len(losers) < num_to_choose - len(winners):
                winners.extend(np.argwhere(remaining).flatten())
                break

            # OK, if we're here, we've been cleared to remove the losers
            remaining[losers] = False
            for ind in losers:
                # Now update the vote ranks.
                offset = np.where(((v > 1) & (v > v[:,ind][:,np.newaxis])),1,0)
                offset[:,ind] = v[:,ind]
                v = v - offset

    # Return the winners
    return winners
            
        
