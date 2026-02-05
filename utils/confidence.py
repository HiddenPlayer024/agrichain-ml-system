def confidence_from_probability(p):
    return round(max(p, 1 - p), 2)
