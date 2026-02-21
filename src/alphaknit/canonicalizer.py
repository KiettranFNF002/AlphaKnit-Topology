"""
Canonicalizer: Normalizes flat token sequences so that
structurally equivalent patterns produce identical output.

Rules:
1. Evenly distribute increases/decreases within a row
2. Consistent spiral direction
3. Standardize start position
"""


def canonicalize_row(row_tokens):
    """
    Re-order tokens within a single row so that increases/decreases
    are evenly spaced among sc stitches.

    Example:
        ['inc', 'sc', 'sc', 'inc', 'sc', 'sc'] (unevenly placed)
        → ['sc', 'inc', 'sc', 'sc', 'inc', 'sc'] (canonical even spacing)

    For a row with N total stitches and K special stitches (inc/dec),
    the canonical form spaces them every N/K positions.
    """
    sc_tokens = [t for t in row_tokens if t == 'sc']
    special_tokens = [t for t in row_tokens if t != 'sc']

    if not special_tokens:
        return list(row_tokens)  # All sc, nothing to reorder

    total = len(row_tokens)
    k = len(special_tokens)
    # Interval between specials
    interval = total // k if k > 0 else total

    result = []
    special_idx = 0
    sc_idx = 0

    for i in range(total):
        # Place a special token at evenly spaced positions
        if special_idx < k and i > 0 and i % interval == interval - 1:
            result.append(special_tokens[special_idx])
            special_idx += 1
        elif sc_idx < len(sc_tokens):
            result.append('sc')
            sc_idx += 1
        elif special_idx < k:
            result.append(special_tokens[special_idx])
            special_idx += 1

    return result


def canonicalize_pattern(flat_tokens):
    """
    Canonicalize a full flat token sequence by:
    1. Keeping mr_N as-is
    2. Splitting remaining tokens into rows (by counting slots consumed)
    3. Canonicalizing each row's token order
    4. Flattening back

    Args:
        flat_tokens: list like ['mr_6', 'sc', 'inc', 'sc', 'inc', ...]

    Returns:
        list of canonicalized tokens
    """
    if not flat_tokens:
        return []

    result = []
    current_row = []
    prev_row_count = 0

    slot_cost = {'sc': 1, 'inc': 1, 'dec': 2, 'slst': 1, 'hdc': 1, 'dc': 1}
    slots_consumed = 0

    for token in flat_tokens:
        # Magic ring: pass through, set initial count
        if token.startswith('mr_'):
            result.append(token)
            try:
                prev_row_count = int(token.split('_')[1])
            except (IndexError, ValueError):
                prev_row_count = 6
            continue

        cost = slot_cost.get(token, 1)

        # Check if adding this token would exceed the row
        if prev_row_count > 0 and slots_consumed + cost > prev_row_count:
            # Flush current row
            canonical_row = canonicalize_row(current_row)
            result.extend(canonical_row)
            prev_row_count = len(current_row)  # New row size = output count of previous
            current_row = []
            slots_consumed = 0

        current_row.append(token)
        slots_consumed += cost

    # Flush last row
    if current_row:
        canonical_row = canonicalize_row(current_row)
        result.extend(canonical_row)

    return result
