from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from .qconfig import QConfig
from torch.jit._recursive import wrap_cpp_module

def _check_is_script_module(model):
    if not isinstance(model, torch.jit.ScriptModule):
        raise ValueError('input must be a script module, got: ' + str(type(model)))

def _check_forward_method(model):
    if not model._c._has_method('forward'):
        raise ValueError('input script module does not have forward method')

def script_qconfig(qconfig):
    return QConfig(
        activation=torch.jit.script(qconfig.activation())._c,
        weight=torch.jit.script(qconfig.weight())._c)

def script_qconfig_dict(qconfig_dict):
    return {k: script_qconfig(v) if v else None for k, v in qconfig_dict.items()}

def _prepare_script(model, qconfig_dict, inplace=False, is_dynamic=False):
    assert not inplace, "The inplace support is still in development"
    _check_is_script_module(model)
    _check_forward_method(model)
    if not all(isinstance(x, str) for x in qconfig_dict.keys()):
        raise ValueError('qconfig_dict should only contain names(str) as keys.')
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    torch._C._jit_pass_dedup_module_uses(model._c)
    model = wrap_cpp_module(torch._C._jit_pass_fold_convbn(model._c))
    return wrap_cpp_module(torch._C._jit_pass_insert_observers(model._c,
                                                               'forward',
                                                               scripted_qconfig_dict,
                                                               inplace,
                                                               is_dynamic))

def prepare_script(model, qconfig_dict, inplace=False):
    return _prepare_script(model, qconfig_dict, inplace, is_dynamic=False)

def prepare_dynamic_script(model, qconfig_dict, inplace=False):
    return _prepare_script(model, qconfig_dict, inplace, is_dynamic=True)

def _convert_script(model, inplace=False, debug=False, is_dynamic=False):
    assert not inplace, "The inplace support is still in development"
    _check_is_script_module(model)
    model.eval()
    model = wrap_cpp_module(torch._C._jit_pass_insert_quant_dequant(model._c, 'forward', inplace, is_dynamic))
    if not debug:
        model = wrap_cpp_module(torch._C._jit_pass_quant_finalize(model._c, is_dynamic))
    return model

def convert_script(model, inplace=False, debug=False):
    return _convert_script(model, inplace, debug, False)

def convert_dynamic_script(model, inplace=False, debug=False):
    return _convert_script(model, inplace, debug, True)

def _quantize_script(model, qconfig_dict, run_fn=None, run_args=None, inplace=False, debug=False, is_dynamic=False):
    assert not inplace, "We don't support inplace right now"
    # Always do inplace convert because the Tensor is already
    # copied in prepare_script when inplace is False
    if is_dynamic:
        model = prepare_dynamic_script(model, qconfig_dict, inplace)
        # TODO: change inplace to True
        model = convert_dynamic_script(model, False, debug)
    else:
        assert run_fn, "Must provide calibration function for post training static quantization"
        assert run_args, "Must provide calibration dataset for post training static quantization"
        model = prepare_script(model, qconfig_dict, inplace)
        run_fn(model, *run_args)
        # TODO: change inplace to True
        model = convert_script(model, False, debug)

    return model

def quantize_script(model, qconfig_dict, run_fn, run_args, inplace=False, debug=False):
    return _quantize_script(model, qconfig_dict, run_fn, run_args, inplace, debug, False)

def quantize_dynamic_script(model, qconfig_dict, inplace=False, debug=False):
    return _quantize_script(model, qconfig_dict, run_args=None, inplace=inplace, debug=debug, is_dynamic=True)
