import logging
import re

import math
import os
import tempfile
from itertools import chain
from pathlib import Path

import amici
import libsbml
import petab.v1 as petab
import sympy as sp
from _collections import OrderedDict
from amici.logging import log_execution_time, set_log_level
from petab.v1.models import MODEL_TYPE_SBML
from sympy.abc import _clash

from . import PREEQ_INDICATOR_ID
from .import_helpers import (
    check_model,
    get_fixed_parameters,
    get_observation_model,
)
from .util import get_states_in_condition_table

logger = logging.getLogger(__name__)


def _workaround_initial_states(
    petab_problem: petab.Problem, sbml_model: libsbml.Model, **kwargs
) -> list[str]:
    """Add initial assignments and their targets to represent initial states
    from the PEtab condition table.

    :return: List of parameters that were added to the model and need to be
        processed as fixed parameters during SBML import.
    """

    # TODO: to parameterize initial states or compartment sizes, we currently
    #  need initial assignments. if they occur in the condition table, we
    #  create a new parameter initial_${speciesOrCompartmentID}.
    #  feels dirty and should be changed (see also #924)

    # state variable IDs and initial values specified via the conditions' table
    initial_states = get_states_in_condition_table(petab_problem)
    # is there any condition that involves preequilibration?
    requires_preequilibration = (
        petab_problem.measurement_df is not None
        and petab.PREEQUILIBRATION_CONDITION_ID in petab_problem.measurement_df
        and petab_problem.measurement_df[petab.PREEQUILIBRATION_CONDITION_ID]
        .notnull()
        .any()
    )
    estimated_parameters_ids = petab_problem.get_x_ids(free=True, fixed=False)
    # any initial states overridden to be estimated via the conditions table?
    has_estimated_initial_states = any(
        par_id in petab_problem.condition_df[initial_states.keys()].values
        for par_id in estimated_parameters_ids
    )

    if (
        has_estimated_initial_states
        and requires_preequilibration
        and kwargs.setdefault("generate_sensitivity_code", True)
    ):
        # To support reinitialization of initial conditions after
        # preequilibration we need fixed parameters for the initial
        # conditions. If we need sensitivities w.r.t. to initial conditions,
        # we need to create non-fixed parameters for the initial conditions.
        # We can't have both for the same state variable.
        # (We could handle it via separate amici models if pre-equilibration
        # and estimation of initial values for a given state variable are
        # used in separate PEtab conditions.)
        # We currently assume that we do need sensitivities w.r.t. initial
        # conditions if sensitivities are needed at all.
        # TODO: check this state by state, then we can support some additional
        #  cases
        raise NotImplementedError(
            "PEtab problems that have both, estimated initial conditions "
            "specified in the condition table, and preequilibration with "
            "initial conditions specified in the condition table are not "
            "supported."
        )

    fixed_parameters = []
    if initial_states and requires_preequilibration:
        # add preequilibration indicator variable
        if sbml_model.getParameter(PREEQ_INDICATOR_ID) is not None:
            raise AssertionError(
                "Model already has a parameter with ID "
                f"{PREEQ_INDICATOR_ID}. Cannot handle "
                "species and compartments in condition table "
                "then."
            )
        indicator = sbml_model.createParameter()
        indicator.setId(PREEQ_INDICATOR_ID)
        indicator.setName(PREEQ_INDICATOR_ID)
        # Can only reset parameters after preequilibration if they are fixed.
        fixed_parameters.append(PREEQ_INDICATOR_ID)
        logger.debug(
            f"Adding preequilibration indicator constant {PREEQ_INDICATOR_ID}"
        )
    logger.debug(
        f"Adding initial assignments for {list(initial_states.keys())}"
    )
    for assignee_id in initial_states:
        init_par_id_preeq = f"initial_{assignee_id}_preeq"
        init_par_id_sim = f"initial_{assignee_id}_sim"
        for init_par_id in (
            [init_par_id_preeq] if requires_preequilibration else []
        ) + [init_par_id_sim]:
            if sbml_model.getElementBySId(init_par_id) is not None:
                raise ValueError(
                    "Cannot create parameter for initial assignment "
                    f"for {assignee_id} because an entity named "
                    f"{init_par_id} exists already in the model."
                )
            init_par = sbml_model.createParameter()
            init_par.setId(init_par_id)
            init_par.setName(init_par_id)
            if requires_preequilibration:
                # must be a fixed parameter to allow reinitialization
                # TODO: also add other initial condition parameters that are
                #  not estimated
                fixed_parameters.append(init_par_id)

        assignment = sbml_model.getInitialAssignment(assignee_id)
        if assignment is None:
            assignment = sbml_model.createInitialAssignment()
            assignment.setSymbol(assignee_id)
        else:
            logger.debug(
                "The SBML model has an initial assignment defined "
                f"for model entity {assignee_id}, but this entity "
                "also has an initial value defined in the PEtab "
                "condition table. The SBML initial assignment will "
                "be overwritten to handle preequilibration and "
                "initial values specified by the PEtab problem."
            )
        if requires_preequilibration:
            formula = (
                f"{PREEQ_INDICATOR_ID} * {init_par_id_preeq} "
                f"+ (1 - {PREEQ_INDICATOR_ID}) * {init_par_id_sim}"
            )
        else:
            formula = init_par_id_sim
        math_ast = libsbml.parseL3Formula(formula)
        assignment.setMath(math_ast)

    return fixed_parameters


def _workaround_observable_parameters(
    observables: dict[str, dict[str, str]],
    sigmas: dict[str, str | float],
    sbml_model: libsbml.Model,
    output_parameter_defaults: dict[str, float] | None,
    jax: bool = False,
) -> None:
    """
    Add PEtab observable parameters to the SBML model.

    The PEtab observable table may contain placeholder parameters that are
    not defined in the SBML model. We need to add them to the SBML model before
    the actual SBML import.
    """
    formulas = chain(
        (val["formula"] for val in observables.values()), sigmas.values()
    )
    output_parameters = OrderedDict()
    for formula in formulas:
        # we want reproducible parameter ordering upon repeated import
        free_syms = sorted(
            sp.sympify(formula, locals=_clash).free_symbols,
            key=lambda symbol: symbol.name,
        )
        for free_sym in free_syms:
            sym = str(free_sym)
            if jax and (m := re.match(r"(noiseParameter\d+)_(\w+)", sym)):
                # group1 is the noise parameter, group2 is the observable, don't add to sbml but replace with generic
                # noise parameter
                sigmas[m.group(2)] = str(
                    sp.sympify(sigmas[m.group(2)], locals=_clash).subs(
                        free_sym, sp.Symbol(m.group(1))
                    )
                )
            elif jax and (
                m := re.match(r"(observableParameter\d+)_(\w+)", sym)
            ):
                # group1 is the noise parameter, group2 is the observable, don't add to sbml but replace with generic
                # observable parameter
                observables[m.group(2)]["formula"] = str(
                    sp.sympify(
                        observables[m.group(2)]["formula"], locals=_clash
                    ).subs(free_sym, sp.Symbol(m.group(1)))
                )
            elif (
                sbml_model.getElementBySId(sym) is None
                and sym != "time"
                and sym not in observables
            ):
                output_parameters[sym] = None
    logger.debug(
        f"Adding output parameters to model: {list(output_parameters.keys())}"
    )
    output_parameter_defaults = output_parameter_defaults or {}
    if extra_pars := (
        set(output_parameter_defaults) - set(output_parameters.keys())
    ):
        raise ValueError(
            f"Default output parameter values were given for {extra_pars}, "
            "but they those are not output parameters."
        )

    for par in output_parameters.keys():
        _add_global_parameter(
            sbml_model=sbml_model,
            parameter_id=par,
            value=output_parameter_defaults.get(par, 0.0),
        )


@log_execution_time("Importing PEtab model", logger)
def import_model_sbml(
    petab_problem: petab.Problem = None,
    model_name: str | None = None,
    model_output_dir: str | Path | None = None,
    verbose: bool | int | None = True,
    allow_reinit_fixpar_initcond: bool = True,
    validate: bool = True,
    non_estimated_parameters_as_constants=True,
    output_parameter_defaults: dict[str, float] | None = None,
    discard_sbml_annotations: bool = False,
    jax: bool = False,
    **kwargs,
) -> amici.SbmlImporter:
    """
    Create AMICI model from PEtab problem

    :param petab_problem:
        PEtab problem.

    :param model_name:
        Name of the generated model. If model file name was provided,
        this defaults to the file name without extension, otherwise
        the SBML model ID will be used.

    :param model_output_dir:
        Directory to write the model code to. Will be created if doesn't
        exist. Defaults to current directory.

    :param verbose:
        Print/log extra information.

    :param allow_reinit_fixpar_initcond:
        See :class:`amici.de_export.ODEExporter`. Must be enabled if initial
        states are to be reset after preequilibration.

    :param validate:
        Whether to validate the PEtab problem

    :param non_estimated_parameters_as_constants:
        Whether parameters marked as non-estimated in PEtab should be
        considered constant in AMICI. Setting this to ``True`` will reduce
        model size and simulation times. If sensitivities with respect to those
        parameters are required, this should be set to ``False``.

    :param output_parameter_defaults:
        Optional default parameter values for output parameters introduced in
        the PEtab observables table, in particular for placeholder parameters.
        dictionary mapping parameter IDs to default values.

    :param discard_sbml_annotations:
        Discard information contained in AMICI SBML annotations (debug).

    :param jax:
        Whether to generate JAX code instead of C++ code.

    :param kwargs:
        Additional keyword arguments to be passed to
        :meth:`amici.sbml_import.SbmlImporter.sbml2amici`.

    :return:
        The created :class:`amici.sbml_import.SbmlImporter` instance.
    """
    from petab.v1.models.sbml_model import SbmlModel

    set_log_level(logger, verbose)

    logger.info("Importing model ...")

    if petab_problem.observable_df is None:
        raise NotImplementedError(
            "PEtab import without observables table "
            "is currently not supported."
        )

    assert isinstance(petab_problem.model, SbmlModel)

    if validate:
        logger.info("Validating PEtab problem ...")
        petab.lint_problem(petab_problem)

    # Model name from SBML ID or filename
    if model_name is None:
        if not (model_name := petab_problem.model.sbml_model.getId()):
            raise ValueError(
                "No `model_name` was provided and no model "
                "ID was specified in the SBML model."
            )

    if model_output_dir is None:
        model_output_dir = os.path.join(
            os.getcwd(), f"{model_name}-amici{amici.__version__}"
        )

    logger.info(
        f"Model name is '{model_name}'.\n"
        f"Writing model code to '{model_output_dir}'."
    )

    # Create a copy, because it will be modified by SbmlImporter
    sbml_doc = petab_problem.model.sbml_model.getSBMLDocument().clone()
    sbml_model = sbml_doc.getModel()

    show_model_info(sbml_model)

    sbml_importer = amici.SbmlImporter(
        sbml_model,
        discard_annotations=discard_sbml_annotations,
    )
    sbml_model = sbml_importer.sbml

    allow_n_noise_pars = (
        not petab.lint.observable_table_has_nontrivial_noise_formula(
            petab_problem.observable_df
        )
    )
    if (
        not jax
        and petab_problem.measurement_df is not None
        and petab.lint.measurement_table_has_timepoint_specific_mappings(
            petab_problem.measurement_df,
            allow_scalar_numeric_noise_parameters=allow_n_noise_pars,
        )
    ):
        raise ValueError(
            "AMICI does not support importing models with timepoint specific "
            "mappings for noise or observable parameters. Please flatten "
            "the problem and try again."
        )

    if petab_problem.observable_df is not None:
        observables, noise_distrs, sigmas = get_observation_model(
            petab_problem.observable_df
        )
    else:
        observables = noise_distrs = sigmas = None

    logger.info(f"Observables: {len(observables)}")
    logger.info(f"Sigmas: {len(sigmas)}")

    if len(sigmas) != len(observables):
        raise AssertionError(
            f"Number of provided observables ({len(observables)}) and sigmas "
            f"({len(sigmas)}) do not match."
        )

    _workaround_observable_parameters(
        observables, sigmas, sbml_model, output_parameter_defaults, jax=jax
    )

    if not jax:
        fixed_parameters = _workaround_initial_states(
            petab_problem=petab_problem,
            sbml_model=sbml_model,
            **kwargs,
        )
    else:
        fixed_parameters = []

    fixed_parameters.extend(
        _get_fixed_parameters_sbml(
            petab_problem=petab_problem,
            non_estimated_parameters_as_constants=non_estimated_parameters_as_constants,
        )
    )

    logger.debug(f"Fixed parameters are {fixed_parameters}")
    logger.info(f"Overall fixed parameters: {len(fixed_parameters)}")
    logger.info(
        "Variable parameters: "
        + str(len(sbml_model.getListOfParameters()) - len(fixed_parameters))
    )

    # Create Python module from SBML model
    if jax:
        sbml_importer.sbml2jax(
            model_name=model_name,
            output_dir=model_output_dir,
            observables=observables,
            sigmas=sigmas,
            noise_distributions=noise_distrs,
            verbose=verbose,
            **kwargs,
        )
        return sbml_importer
    else:
        sbml_importer.sbml2amici(
            model_name=model_name,
            output_dir=model_output_dir,
            observables=observables,
            constant_parameters=fixed_parameters,
            sigmas=sigmas,
            allow_reinit_fixpar_initcond=allow_reinit_fixpar_initcond,
            noise_distributions=noise_distrs,
            verbose=verbose,
            **kwargs,
        )

    if kwargs.get(
        "compile",
        amici._get_default_argument(sbml_importer.sbml2amici, "compile"),
    ):
        # check that the model extension was compiled successfully
        model_module = amici.import_model_module(model_name, model_output_dir)
        model = model_module.getModel()
        check_model(amici_model=model, petab_problem=petab_problem)

    return sbml_importer


def show_model_info(sbml_model: "libsbml.Model"):
    """Log some model quantities"""

    logger.info(f"Species: {len(sbml_model.getListOfSpecies())}")
    logger.info(
        "Global parameters: " + str(len(sbml_model.getListOfParameters()))
    )
    logger.info(f"Reactions: {len(sbml_model.getListOfReactions())}")


# TODO - remove?!
def species_to_parameters(
    species_ids: list[str], sbml_model: "libsbml.Model"
) -> list[str]:
    """
    Turn a SBML species into parameters and replace species references
    inside the model instance.

    :param species_ids:
        list of SBML species ID to convert to parameters with the same ID as
        the replaced species.

    :param sbml_model:
        SBML model to modify

    :return:
        list of IDs of species which have been converted to parameters
    """
    transformables = []

    for species_id in species_ids:
        species = sbml_model.getSpecies(species_id)

        if species.getHasOnlySubstanceUnits():
            logger.warning(
                f"Ignoring {species.getId()} which has only substance units."
                " Conversion not yet implemented."
            )
            continue

        if math.isnan(species.getInitialConcentration()):
            logger.warning(
                f"Ignoring {species.getId()} which has no initial "
                "concentration. Amount conversion not yet implemented."
            )
            continue

        transformables.append(species_id)

    # Must not remove species while iterating over getListOfSpecies()
    for species_id in transformables:
        species = sbml_model.removeSpecies(species_id)
        par = sbml_model.createParameter()
        par.setId(species.getId())
        par.setName(species.getName())
        par.setConstant(True)
        par.setValue(species.getInitialConcentration())
        par.setUnits(species.getUnits())

    # Remove from reactants and products
    for reaction in sbml_model.getListOfReactions():
        for species_id in transformables:
            # loop, since removeX only removes one instance
            while reaction.removeReactant(species_id):
                # remove from reactants
                pass
            while reaction.removeProduct(species_id):
                # remove from products
                pass
            while reaction.removeModifier(species_id):
                # remove from modifiers
                pass

    return transformables


def _add_global_parameter(
    sbml_model: libsbml.Model,
    parameter_id: str,
    parameter_name: str = None,
    constant: bool = False,
    units: str = "dimensionless",
    value: float = 0.0,
) -> libsbml.Parameter:
    """Add new global parameter to SBML model

    Arguments:
        sbml_model: SBML model
        parameter_id: ID of the new parameter
        parameter_name: Name of the new parameter
        constant: Is parameter constant?
        units: SBML unit ID
        value: parameter value

    Returns:
        The created parameter
    """
    if parameter_name is None:
        parameter_name = parameter_id

    p = sbml_model.createParameter()
    p.setId(parameter_id)
    p.setName(parameter_name)
    p.setConstant(constant)
    p.setValue(value)
    p.setUnits(units)
    return p


def _get_fixed_parameters_sbml(
    petab_problem: petab.Problem,
    non_estimated_parameters_as_constants=True,
) -> list[str]:
    """
    Determine, set and return fixed model parameters.

    Non-estimated parameters and parameters specified in the condition table
    are turned into constants (unless they are overridden).
    Only global SBML parameters are considered. Local parameters are ignored.

    :param petab_problem:
        The PEtab problem instance

    :param non_estimated_parameters_as_constants:
        Whether parameters marked as non-estimated in PEtab should be
        considered constant in AMICI. Setting this to ``True`` will reduce
        model size and simulation times. If sensitivities with respect to those
        parameters are required, this should be set to ``False``.

    :return:
        list of IDs of parameters which are to be considered constant.
    """
    if not petab_problem.model.type_id == MODEL_TYPE_SBML:
        raise ValueError("Not an SBML model.")
    # initial concentrations for species or initial compartment sizes in
    # condition table will need to be turned into fixed parameters

    # if there is no initial assignment for that species, we'd need
    # to create one. to avoid any naming collision right away, we don't
    # allow that for now

    # we can't handle them yet
    compartments = [
        col
        for col in petab_problem.condition_df
        if petab_problem.model.sbml_model.getCompartment(col) is not None
    ]
    if compartments:
        raise NotImplementedError(
            "Can't handle initial compartment sizes "
            "at the moment. Consider creating an "
            f"initial assignment for {compartments}"
        )

    fixed_parameters = get_fixed_parameters(
        petab_problem, non_estimated_parameters_as_constants
    )

    # exclude targets of rules or initial assignments that are not numbers
    sbml_model = petab_problem.model.sbml_model
    parser_settings = libsbml.L3ParserSettings()
    parser_settings.setModel(sbml_model)
    parser_settings.setParseUnits(libsbml.L3P_NO_UNITS)

    for fixed_parameter in fixed_parameters.copy():
        # check global parameters
        if sbml_model.getRuleByVariable(fixed_parameter):
            fixed_parameters.remove(fixed_parameter)
            continue
        if ia := sbml_model.getInitialAssignmentBySymbol(fixed_parameter):
            sym_math = sp.sympify(
                libsbml.formulaToL3StringWithSettings(
                    ia.getMath(), parser_settings
                )
            )
            if not sym_math.evalf().is_Number:
                fixed_parameters.remove(fixed_parameter)
                continue

    return list(sorted(fixed_parameters))


def _create_model_output_dir_name(
    sbml_model: "libsbml.Model",
    model_name: str | None = None,
    jax: bool = False,
) -> Path:
    """
    Find a folder for storing the compiled amici model.
    If possible, use the sbml model id, otherwise create a random folder.
    The folder will be located in the `amici_models` subfolder of the current
    folder.
    """
    BASE_DIR = Path("amici_models").absolute()
    BASE_DIR.mkdir(exist_ok=True)
    # try model_name
    suffix = "_jax" if jax else ""
    if model_name:
        return BASE_DIR / (model_name + suffix)

    # try sbml model id
    if sbml_model_id := sbml_model.getId():
        return BASE_DIR / (sbml_model_id + suffix)

    # create random folder name
    return Path(tempfile.mkdtemp(dir=BASE_DIR))
