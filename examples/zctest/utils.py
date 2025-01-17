
from pathlib import Path
import os
import re
import uuid

import onnxruntime as ort
import onnx
import numpy as np
from onnxruntime.quantization import CalibrationDataReader


class ListDataReader(CalibrationDataReader):
    def __init__(self, data_list, datasize=0) -> None:
        self.index = 0
        self.datasize = len(data_list) if datasize <= 0 else min(len(data_list), datasize)
        self.data_list = data_list[:self.datasize]

    def get_next(self):
        if self.index < len(self.data_list):
            data = self.data_list[self.index]
            self.index += 1
            return data
        else:
            return None
        
    def rewind(self) -> None:
        self.index = 0


def make_cpu_session(model_path):
    session = ort.InferenceSession(model_path)
    return session

def make_qnn_session(model_path, verbose=False, use_context=False, qnn_cpu=False):
    options = ort.SessionOptions()
    if use_context and not qnn_cpu:
        onnx_ctx_path = Path(model_path).with_suffix('.onnx_ctx.onnx')
        if os.path.exists(onnx_ctx_path):
            print('using ctx')
            model_path = onnx_ctx_path
        else:
            options.add_session_config_entry('ep.context_enable', '1')
            options.add_session_config_entry('ep.context_embed_mode', '0')
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
    if verbose:
        options.log_severity_level = 0
    backend_path = "QnnHtp.dll" if not qnn_cpu else "QnnCpu.dll"
    session = ort.InferenceSession(model_path, 
                                        sess_options=options,
                                        providers=["QNNExecutionProvider"],
                                        provider_options=[
                                            {
                                                'profiling_level': 'detailed',   # Enable detailed profiling
                                                'profiling_file_path': 'qnn_profile.csv',  # Path to save profiling data
                                                "backend_path": backend_path,
                                                "htp_graph_finalization_optimization_mode": "3",
                                                "htp_performance_mode": "burst",
                                                "offload_graph_io_quantization": "1",  # a tiny bit faster?
                                             }
                                        ]
    )
    return session

def make_session(model_path, ep, **kwargs):
    if ep == 'qnn':
        return make_qnn_session(model_path, **kwargs)
    elif ep == 'cpu':
        return make_cpu_session(model_path)
    else:
        raise ValueError(f"Unknown execution provider {ep}")

def add_outputs(model, new_outputs):
    inputs = {input.name: input for input in model.graph.input}
    outputs = {output.name: output for output in model.graph.output}
    value_infos = {value_info.name: value_info for value_info in model.graph.value_info}
    all_value_infos = dict()
    all_value_infos.update(inputs)
    all_value_infos.update(outputs)
    all_value_infos.update(value_infos)

    new_model = onnx.ModelProto()
    new_model.CopyFrom(model)

    new_model.graph.output.extend([all_value_infos[name] for name in new_outputs])

    return new_model

def add_all_to_outputs(model):
    # add all inputs and value_infos to outputs, so we can infer all shapes by running an exmaple
    new_model = onnx.ModelProto()
    new_model.CopyFrom(model)
    new_model.graph.output.extend(model.graph.input)
    new_model.graph.output.extend(model.graph.value_info)
    return new_model, [info.name for info in new_model.graph.output]


def apply_shape_map_(model, shape_map):
    # in-place; shape_map is a dict of all shapes of model graph
    for infos in (model.graph.input, model.graph.output, model.graph.value_info):
        for info in infos:
            assert info.name in shape_map, f"shape of {info.name} not found in shape_map"
            assert len(info.type.tensor_type.shape.dim) == len(shape_map[info.name]), \
                f"dimension of {info.name} not match shape_map"
            for i, size in enumerate(shape_map[info.name]):
                info.type.tensor_type.shape.dim[i].Clear()
                info.type.tensor_type.shape.dim[i].dim_value = size
    return model


def get_shape(name, value_infos, initializers):
    if name in value_infos:
        value_info = value_infos[name]
        return np.array([dim.dim_value for dim in value_info.type.tensor_type.shape.dim], dtype=np.int64)
    elif name in initializers:
        initializer = initializers[name]
        return np.array(initializer.dims, dtype=np.int64)


def make_dynamic_shapes_fixed_by_sample(input_model, output_model, sample_inputs):
    model = onnx.load(input_model)
    all_output_model, output_names = add_all_to_outputs(model)
    output_dir = Path(output_model).parent
    name = str(uuid.uuid4())
    all_output_model_path = str(output_dir / f"model_all_outputs_{name}.onnx")
    all_output_model_data_path = str(output_dir / f"model_all_outputs_{name}.onnx_data")
    onnx.save(
        all_output_model,
        all_output_model_path,
        save_as_external_data=True,
        location=f"model_all_outputs_{name}.onnx_data",
    )
    session = ort.InferenceSession(all_output_model_path)
    outputs = session.run(output_names, sample_inputs)
    shape_map = {name: output.shape for name, output in zip(output_names, outputs)}
    try:
        os.remove(all_output_model_path)
    except FileNotFoundError:
        pass
    try:
        os.remove(all_output_model_data_path)
    except FileNotFoundError:
        pass
    apply_shape_map_(model, shape_map)
    onnx.save(model, output_model, save_as_external_data=True)


def fold_layernorms(
        input_model,
        output_model,
        layernorm_keyword='norm',
        epsilon=1e-5,  # TODO: try different values, value for float32 model is is 1e-6
    ):
    # layer norm ops are usually converted to primitive math ops (Add, Mul, Div, Sqrt etc.)
    # which may have bad numerical behaviors when run by QNN EP ()
    # this function folds these primitive ops back to LayerNormalization op, which is supported by QNN EP
    # quick tests suggest that this folding fixes some models' misbehaviors under QNN EP
    model = onnx.load(input_model)
    nodes = {node.name: node for node in model.graph.node}
    layernorm_groups = dict()

    for key in nodes.keys():
        if layernorm_keyword in key:
            prefix = key[:(key.index(layernorm_keyword) + len(layernorm_keyword))]
            if prefix not in layernorm_groups:
                layernorm_groups[prefix] = []
            layernorm_groups[prefix].append(key)

    nodes_to_remove = set()
    for group in layernorm_groups.values():
        nodes_to_remove.update(group)

    nodes_to_replace = dict()

    for prefix in layernorm_groups.keys():
        pow_name = f'{prefix}/Pow'
        pow_node = nodes[pow_name]
        input_name = pow_node.input[0]
        mul1_name = f'{prefix}/Mul_1'
        mul1_node = nodes[mul1_name]
        output_name = mul1_node.output[0]
        scale_name = mul1_node.input[0]

        layer_norm_node = onnx.helper.make_node(
            'LayerNormalization',
            inputs=[input_name, scale_name],
            outputs=[output_name],
            axis=-1,
            epsilon=epsilon,
            stash_type=1,
            name = prefix + '/LayerNormalization',
            # domain ?
        )
        nodes_to_replace[prefix + '/Pow'] = layer_norm_node

    new_nodes = list()
    for node in model.graph.node:
        if node.name in nodes_to_replace:
            new_nodes.append(nodes_to_replace[node.name])
        elif node.name not in nodes_to_remove:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    onnx_data_name = Path(output_model).with_suffix('.onnx_data').name
    onnx.save(
        model,
        output_model,
        save_as_external_data=True,
        location=onnx_data_name
    )

# def fold_gemms(
#         input_model,
#         output_model,
#     ):
#     # Gemm supports 2D inputs A, B only
#     matmul_output_pattern = re.compile(r'MatMul(_\d+)?_output_\d+')
#     add_type = 'Add'

#     model = onnx.load(input_model)
#     nodes = {node.name: node for node in model.graph.node}
#     output_to_node_map = {
#         output_name: node.name
#         for node in model.graph.node
#         for output_name in node.output
#     }

#     nodes_to_replace = dict()
#     nodes_to_remove = set()

#     for name, node in nodes.items():
#         if node.op_type != add_type:
#             continue

#         add_node = node
#         add_name = name
#         matmul_output_name = None
#         for input_name in node.input:
#             if matmul_output_pattern.match(os.path.basename(input_name)):
#                 assert matmul_output_name is None
#                 matmul_output_name = input_name
#         if matmul_output_name is not None:
#             matmul_name = output_to_node_map[matmul_output_name]
#             matmul_node = nodes[matmul_name]
#             add_output_name = add_node.output[0]
#             gemm_input_names = [input_name for input_name in matmul_node.input]
#             for add_input_name in add_node.input:
#                 if add_input_name == matmul_output_name:
#                     pass
#                 else:
#                     gemm_input_names.append(add_input_name)
#             # print(add_name, matmul_name, gemm_input_names)

#             gemm_node = onnx.helper.make_node(
#                 'Gemm',
#                 inputs=gemm_input_names,
#                 outputs=[add_output_name],
#                 name=matmul_name.replace('MatMul', 'Gemm'),
#                 # domain ?
#             )
#             nodes_to_replace[matmul_name] = gemm_node
#             nodes_to_remove.add(add_name)

#     new_nodes = list()
#     for node in model.graph.node:
#         if node.name in nodes_to_replace:
#             new_nodes.append(nodes_to_replace[node.name])
#         elif node.name not in nodes_to_remove:
#             new_nodes.append(node)

#     del model.graph.node[:]
#     model.graph.node.extend(new_nodes)

#     onnx_data_name = Path(output_model).with_suffix('.onnx_data').name
#     onnx.save(
#         model,
#         output_model,
#         save_as_external_data=True,
#         location=onnx_data_name
#     )

def remove_cast_f32(
        input_model,
        output_model,
        keyword=None, # keyword to match cast nodes
    ):
    cast_type = 'Cast'
    cast_to_dtype = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(np.float32))


    model = onnx.load(input_model)
    nodes = {node.name: node for node in model.graph.node}
    output_to_node_map = {
        output_name: node.name
        for node in model.graph.node
        for output_name in node.output
    }

    nodes_to_remove = set()

    for node in model.graph.node:
        if node.op_type == cast_type and (keyword is None or keyword in node.name):
            assert len(node.attribute) == 1
            assert node.attribute[0].name == 'to'
            if node.attribute[0].i == cast_to_dtype:
                nodes_to_remove.add(node.name)
                input_name = node.input[0]
                output_name = node.output[0]
                prev_node = nodes[output_to_node_map[input_name]]
                prev_node_outputs = [name if name != input_name else output_name for name in prev_node.output]
                del prev_node.output[:]
                prev_node.output.extend(prev_node_outputs)

    new_nodes = [node for node in model.graph.node if node.name not in nodes_to_remove]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    onnx_data_name = Path(output_model).with_suffix('.onnx_data').name
    onnx.save(
        model,
        output_model,
        save_as_external_data=True,
        location=onnx_data_name
    )

def replace_expand_with_tile(
        input_model,
        output_model,
    ):
    expand_type = 'Expand'

    model = onnx.load(input_model)
    nodes = {node.name: node for node in model.graph.node}
    value_infos = {value_info.name: value_info for value_info in model.graph.value_info}
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}

    nodes_to_replace = dict()
    initializers_to_add = list()

    for node in nodes.values():
        if node.op_type == expand_type:
            # print(node.name)
            input_shape = get_shape(node.input[0], value_infos, initializers)
            output_shape = get_shape(node.output[0], value_infos, initializers)
            # print(input_shape, output_shape)
            if input_shape.shape != output_shape.shape:
                # print('dim not match, need unsqueeze')
                continue
            assert np.all(output_shape % input_shape == 0)
            repeats_data = output_shape // input_shape
            repeats = onnx.helper.make_tensor(
                node.name + '_repeats',
                onnx.TensorProto.INT64,
                repeats_data.shape,
                repeats_data
            )
            initializers_to_add.append(repeats)
            tile_node = onnx.helper.make_node(
                'Tile',
                inputs=[node.input[0], repeats.name],
                outputs=[node.output[0]],
                name=node.name
            )
            nodes_to_replace[node.name] = tile_node

    new_nodes = list()
    for node in model.graph.node:
        if node.name in nodes_to_replace:
            new_nodes.append(nodes_to_replace[node.name])
        else:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(initializers_to_add)
    onnx_data_name = Path(output_model).with_suffix('.onnx_data').name
    onnx.save(
        model,
        output_model,
        save_as_external_data=True,
        location=onnx_data_name
    )