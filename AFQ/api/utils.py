from importlib import import_module
from AFQ.viz.utils import viz_import_msg_error
from AFQ.utils.docstring_parser import parse_numpy_docstring
import immlib
import logging
import inspect

from dipy.io.stateful_tractogram import set_sft_logger_level


__all__ = [
    "methods_descriptors", "kwargs_descriptors", "AFQclass_doc"]


set_sft_logger_level(logging.CRITICAL)


task_modules = [
    "structural", "data", "tissue", "mapping",
    "segmentation", "tractography", "viz"]

methods_descriptors = {
    "dwi_data_file": "Path to DWI data file",
    "bval_file": "Path to bval file",
    "bvec_file": "Path to bvec file",
    "output_dir": "Path to output directory",
    "best_scalar": "Go-to scalar for visualizations",
    "base_fname": "Base file name for outputs",
    "pve_csf": "Cerebrospinal fluid partial volume estimate map",
    "pve_gm": "Gray matter partial volume estimate map",
    "pve_wm": "White matter partial volume estimate map",
}

methods_sections = {
    "dwi_data_file": "data",
    "bval_file": "data",
    "bvec_file": "data",
    "t1_file": "data",
    "output_dir": "data",
    "best_scalar": "tractography",
    "base_fname": "data",
    "pve_csf": "tractography",
    "pve_gm": "tractography",
    "pve_wm": "tractography",
}

# These kwargs are used to constrcut the plan, not in the plan
# so we don't want to warn if they are unused in the plan
used_kwargs_exceptions = [
    "pve",
    "reg_subject_spec",
    "import_tract",
    "brain_mask_definition"]

kwargs_descriptors = {}
for task_module in task_modules:
    kwargs_descriptors[task_module] = {}
    for calc_obj in import_module(
            f"AFQ.tasks.{task_module}").__dict__.values():
        if immlib.is_calcfn(calc_obj):
            docstr_parsed = parse_numpy_docstring(calc_obj)
            if len(calc_obj.calc.outputs) > 1:
                eff_descs = docstr_parsed["description"].split(",")
                if len(eff_descs) != len(calc_obj.calc.outputs):
                    raise NotImplementedError((
                        "If calc method has mutliple outputs, "
                        "their descriptions must be divided by commas."
                        f" {calc_obj} has {len(eff_descs)} comma-divided"
                        f"sections but {len(calc_obj.calc.outputs)} outputs"))
                for ii in range(len(calc_obj.calc.outputs)):
                    if eff_descs[ii][0] in [' ', '\n']:
                        eff_descs[ii] = eff_descs[ii][1:]
                    if eff_descs[ii][:3] == "and":
                        eff_descs[ii] = eff_descs[ii][3:]
                    if eff_descs[ii][0] in [' ', '\n']:
                        eff_descs[ii] = eff_descs[ii][1:]
                    methods_descriptors[
                        calc_obj.calc.outputs[ii]] =\
                        eff_descs[ii]
                    methods_sections[calc_obj.calc.outputs[ii]] =\
                        task_module
            else:
                methods_descriptors[
                    calc_obj.calc.outputs[0]] =\
                    docstr_parsed["description"]
                methods_sections[calc_obj.calc.outputs[0]] =\
                    task_module
            sig = inspect.signature(calc_obj)
            for arg, info in docstr_parsed["arguments"].items():
                param = sig.parameters.get(arg)
                if "help" in info:
                    kwargs_descriptors[task_module][arg] = dict(
                        desc=info["help"],
                        kind=info["metavar"],
                        default=param.default)
                if arg not in methods_sections:
                    methods_sections[arg] = task_module


AFQclass_doc = (
    "Here are the arguments you can pass to kwargs,"
    " to customize the tractometry pipeline. They are organized"
    " into 5 sections.\n")
for task_module in task_modules:
    AFQclass_doc = AFQclass_doc + "\n"
    AFQclass_doc = AFQclass_doc +\
        "==========================================================\n"
    AFQclass_doc = AFQclass_doc + task_module.upper() + "\n"
    AFQclass_doc = AFQclass_doc +\
        "==========================================================\n"
    for arg, info in kwargs_descriptors[task_module].items():
        AFQclass_doc = AFQclass_doc + arg + ": " + info["kind"]
        AFQclass_doc = AFQclass_doc + "\n\t"
        AFQclass_doc = AFQclass_doc + info["desc"].replace(
            "\n", "\n\t")
        AFQclass_doc = AFQclass_doc + "\n\n"


valid_exports_string = (
    "Here is a list of valid attributes "
    f"to export: {methods_sections.keys()}")


def check_attribute(attr_name):
    if attr_name == "help":
        print(valid_exports_string)
        return False

    if attr_name[:-5] in task_modules:
        return None

    if attr_name in methods_sections:
        return f"{methods_sections[attr_name]}_imap"

    raise ValueError(
        f"{attr_name} not found for export. {valid_exports_string}")


def export_all_helper(api_afq_object, xforms, indiv, viz):
    if xforms:
        try:
            api_afq_object.export("b0_warped")
        except Exception as e:
            api_afq_object.logger.warning((
                "Failed to export warped b0. This could be because your "
                "mapping type is only compatible with transformation "
                f"from template to subject space. The error is: {e}"))
        api_afq_object.export("template_xform")

    if indiv:
        api_afq_object.export("indiv_bundles")
        api_afq_object.export("rois")
    api_afq_object.export("sl_counts")
    api_afq_object.export("median_bundle_lengths")
    api_afq_object.export("profiles")

    if viz:
        try:
            import pingouin
            import seaborn
            import IPython
        except (ImportError, ModuleNotFoundError):
            api_afq_object.logger.warning(viz_import_msg_error("plot"))
        else:
            api_afq_object.export("tract_profile_plots")
        api_afq_object.export("all_bundles_figure")
        api_afq_object.export("indiv_bundles_figures")
