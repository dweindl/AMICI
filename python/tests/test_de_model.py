import pytest
import sympy as sp
from amici._symbolic.de_model_components import Event, FreeParameter
from amici.importers.utils import amici_time_symbol
from amici.testing import skip_on_valgrind


@skip_on_valgrind
def test_model_quantity_reserved_name():
    """`t` and the fixed array-parameter names (x, p, k, h, w, y) are
    reserved for ModelQuantity symbols: the JAX exporter has no mangling of
    its own and relies on these being renamed before codegen ever sees
    them (unlike the C++ backend, whose printer mangles every identifier).
    A name that only collides with a C++ keyword/macro (e.g. NULL) is
    handled by that mangling instead and doesn't need to be rejected
    here."""
    FreeParameter(symbol=sp.Symbol("NULL"), name="NULL", value=1.0)

    for name in ("t", "x", "p", "k", "h", "w", "y"):
        with pytest.raises(ValueError, match="Cannot add"):
            FreeParameter(symbol=sp.Symbol(name), name=name, value=1.0)


@skip_on_valgrind
def test_event_trigger_time():
    e = Event(
        symbol=sp.Symbol("event1"),
        name="event name",
        value=amici_time_symbol - 10,
        assignments=sp.Float(1),
        use_values_from_trigger_time=False,
    )
    assert e.triggers_at_fixed_timepoint() is True
    assert e.get_trigger_time() == 10

    # fixed, but multiple timepoints - not (yet) supported
    e = Event(
        symbol=sp.Symbol("event1"),
        name="event name",
        value=sp.sin(amici_time_symbol),
        assignments=sp.Float(1),
        use_values_from_trigger_time=False,
    )
    assert e.triggers_at_fixed_timepoint() is False

    e = Event(
        symbol=sp.Symbol("event1"),
        name="event name",
        value=amici_time_symbol / 2,
        assignments=sp.Float(1),
        use_values_from_trigger_time=False,
    )
    assert e.triggers_at_fixed_timepoint() is True
    assert e.get_trigger_time() == 0

    # parameter-dependent triggers - not (yet) supported
    e = Event(
        symbol=sp.Symbol("event1"),
        name="event name",
        value=amici_time_symbol - sp.Symbol("delay"),
        assignments=sp.Float(1),
        use_values_from_trigger_time=False,
    )
    assert e.triggers_at_fixed_timepoint() is False
