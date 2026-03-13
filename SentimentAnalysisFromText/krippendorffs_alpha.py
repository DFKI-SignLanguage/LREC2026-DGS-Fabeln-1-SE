import numpy as np
from itertools import combinations

def _as_numpy(a):
    return np.array(a, dtype=object)

def _is_nan(x):
    # Works for None and np.nan
    try:
        return x is None or (isinstance(x, float) and np.isnan(x))
    except:
        return False

def _valid_pairs(values):
    # Return all index pairs (i,j) with non-missing values
    idx = [i for i, v in enumerate(values) if not _is_nan(v)]
    return [(i, j) for i, j in combinations(idx, 2)]

def _metric_nominal(u, v):
    return 0.0 if u == v else 1.0

def _metric_interval(u, v):
    # Squared difference
    return float(u - v) ** 2

def _metric_ratio(u, v):
    # Squared relative difference; treat 0/0 as 0, x/0 as large
    u = float(u); v = float(v)
    if u == v:
        return 0.0
    if u == 0.0 and v == 0.0:
        return 0.0
    denom = (abs(u) + abs(v))
    return ((u - v) / denom) ** 2 if denom != 0 else np.inf

def _ordinal_distance(u, v, categories, weights=None):
    # Krippendorff ordinal distance uses cumulative proportions between ranks.
    # This implementation follows the cumulative rank distance squared.
    # categories: ordered unique values
    pos = {c:i for i, c in enumerate(categories)}
    iu, iv = pos[u], pos[v]
    if iu == iv:
        return 0.0
    # default equal spacing
    if weights is None:
        weights = np.ones(len(categories) - 1)
    # cumulative sum between iu and iv
    lo, hi = sorted((iu, iv))
    return float(np.sum(weights[lo:hi]) ** 2)

def krippendorff_alpha(data, level='nominal', categories=None, ordinal_weights=None):
    """
    Compute Krippendorff's alpha.
    
    Parameters
    -  array-like (n_items, n_annotators); entries are labels or ratings. Missing as None or np.nan.
    - level: 'nominal', 'ordinal', 'interval', or 'ratio'.
    - categories: for nominal/ordinal, optional iterable of allowed categories.
                  If None, inferred from observed (non-missing) values.
    - ordinal_weights: for ordinal distance spacing between adjacent ranks (len = K-1). Optional.
    
    Returns
    - alpha: float (NaN if cannot be computed, e.g., no pairwise overlap)
    """
    A = _as_numpy(data)
    if A.ndim != 2:
        raise ValueError("data must be a 2D array: shape (n_items, n_annotators)")
        
    # Choose distance function
    if level == 'nominal':
        def delta(u, v): return _metric_nominal(u, v)
    elif level == 'interval':
        def delta(u, v): return _metric_interval(float(u), float(v))
    elif level == 'ratio':
        def delta(u, v): return _metric_ratio(float(u), float(v))
    elif level == 'ordinal':
        # categories must be known/inferred and ordered
        # infer categories in sorted order if not provided
        obs = [x for x in A.flatten() if not _is_nan(x)]
        if len(obs) == 0:
            return np.nan
        if categories is None:
            # sort unique values as observed order
            try:
                categories = sorted(set(obs))
            except TypeError:
                # If values aren’t directly sortable, fallback to observed order
                seen = []
                for x in obs:
                    if x not in seen:
                        seen.append(x)
                categories = seen
        cats = list(categories)
        def delta(u, v): 
            return _ordinal_distance(u, v, cats, ordinal_weights)
    else:
        raise ValueError("level must be one of {'nominal','ordinal','interval','ratio'}")
    
    # Helper to compute disagreement within an item (Do numerator component)
    def item_disagreement(values):
        pairs = _valid_pairs(values)
        if not pairs:
            return 0.0, 0  # no pairs
        dsum = 0.0
        m = 0
        for i, j in pairs:
            ui, vj = values[i], values[j]
            dsum += delta(ui, vj)
            m += 1
        return dsum, m
    
    # Observed disagreement Do
    Do_num = 0.0
    Do_den = 0
    for i in range(A.shape[0]):
        row = list(A[i])
        dsum, m = item_disagreement(row)
        Do_num += dsum
        Do_den += m
    
    if Do_den == 0:
        # No overlapping annotations across items
        return np.nan
    Do = Do_num / Do_den
    
    # Expected disagreement De
    # Build pooled value distribution (handling scale-specific needs)
    vals = [x for x in A.flatten() if not _is_nan(x)]
    if len(vals) == 0:
        return np.nan
    
    if level in ('nominal', 'ordinal'):
        # Use category frequencies
        if categories is None:
            # already inferred for ordinal; for nominal infer here
            categories = sorted(set(vals)) if level == 'nominal' else categories
        cats = list(categories)
        counts = {c:0 for c in cats}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        N = sum(counts.values())
        if N < 2:
            return np.nan
        # Expected disagreement over all unordered pairs drawn from the pooled distribution
        # De = sum_{u<v} 2 * p(u) p(v) * delta(u,v) + sum_{u} p(u)p(u)*delta(u,u)=0 => only off-diagonals
        # Efficiently: De = sum_{u} sum_{v} p(u)p(v) delta(u,v)
        probs = {c: counts[c] / N for c in cats}
        De = 0.0
        for u in cats:
            for v in cats:
                De += probs[u] * probs[v] * delta(u, v)
    else:
        # interval/ratio: use all observed values with empirical probabilities
        unique, freqs = np.unique(np.array(vals, dtype=float), return_counts=True)
        N = freqs.sum()
        if N < 2:
            return np.nan
        p = freqs / N
        De = 0.0
        for i, u in enumerate(unique):
            for j, v in enumerate(unique):
                De += p[i] * p[j] * delta(u, v)
    
    if De == 0:
        # Perfect homogeneity in the pooled distribution => alpha undefined; treat as 1 if Do==0
        return 1.0 if Do == 0 else np.nan
    
    alpha = 1.0 - (Do / De)
    return alpha

#
# Example usage for 4 annotators on nominal labels with missing values:
if __name__ == "__main__":

    data = [
        ['A', 'A', 'B', 'A'],
        ['B', 'B', None, 'B'],
        ['C', 'C', 'C', 'C'],
        ['A', None, 'A', 'A'],
        ['B', 'A', 'B', 'B'],
    ]
    print("Alpha (nominal):", krippendorff_alpha(data, level='nominal'))

    # Ordinal example
    data_ord = [
        [1, 1, 2, 1],
        [2, 2, 2, None],
        [3, 3, 3, 3],
        [1, None, 1, 2],
    ]
    print("Alpha (ordinal):", krippendorff_alpha(data_ord, level='ordinal'))
    print("Alpha (nominal on the same ordinal data):", krippendorff_alpha(data_ord, level='nominal'))

    # Interval example
    data_int = [
        [2.1, 2.0, 2.3, np.nan],
        [1.0, 1.1, 1.2, 1.1],
        [3.4, 3.3, 3.5, 3.4],
    ]
    print("Alpha (interval):", krippendorff_alpha(data_int, level='interval'))
