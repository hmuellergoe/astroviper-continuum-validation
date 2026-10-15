"""Translate CASA's inclusive CLEAN limits to AstroVIPER iteration budgets."""


def casa_to_astroviper_iteration_limit(limit: int) -> int:
    """Return the AstroVIPER budget equivalent to one CASA iteration limit."""
    limit = int(limit)
    return limit + 1 if limit > 0 else limit
