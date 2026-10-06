"""CSV loader preserving large neuron IDs; explicit transmitter assumptions."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from neural_model import NeuralModel


def normalized(text):
    return re.sub(r'[^a-z0-9]', '', str(text).lower())


def column(frame, requested, aliases, required=True):
    if requested:
        if requested not in frame.columns:
            raise ValueError(f'Column {requested!r} missing. Available: {list(frame.columns)}')
        return requested
    matches = [c for c in frame.columns if normalized(c) in {normalized(a) for a in aliases}]
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    raise ValueError(f'Set column explicitly in config. Candidates: {matches}; headers: {list(frame.columns)}')


def checked_ids(series):
    result = series.astype(str).str.strip()
    if not result.str.fullmatch(r'\d+').all():
        raise ValueError('BANC IDs must be digit strings. Scientific notation/decimal IDs are rejected.')
    return result


def load_network(cfg, folder, demo=False):
    if demo:
        ids = ['demo_sensory_L', 'demo_sensory_R', 'demo_output_L', 'demo_output_R']
        populations = dict(input_left=[ids[0]], input_right=[ids[1]],
                           output_left=[ids[2]], output_right=[ids[3]])
        model = NeuralModel(ids, [0, 1], [2, 3], [40, 40], ['exc'] * 4, cfg['neural'])
        return model, populations, {'mode': 'SYNTHETIC DEMO', 'neurons': 4, 'edges': 2}
    settings, columns = cfg['data'], cfg['columns']
    for name in ('input_left', 'input_right', 'output_left', 'output_right'):
        values = cfg['populations'][name]
        if not values or any(not isinstance(v, str) for v in values):
            raise ValueError(f'Populate {name} in hybrid_config.json with quoted BANC ID strings. '
                             'Demo mode works without these; BANC IDs are never guessed.')
    attrs_path = (folder / settings['neurons_csv']).resolve()
    conn_path = (folder / settings['connections_csv']).resolve()
    attrs = pd.read_csv(attrs_path, dtype=str, keep_default_na=False)
    idcol = column(attrs, columns['neuron_id'], ['root_id', 'Root ID', 'banc_888_id'])
    attrs[idcol] = checked_ids(attrs[idcol])
    if attrs[idcol].duplicated().any():
        raise ValueError('Duplicate IDs in neuron attributes. Resolve these before loading.')
    verified = column(attrs, columns['verified_nt'], ['Verified NT type', 'nt_type_verified', 'verified_nt'], False)
    predicted = column(attrs, columns['predicted_nt'], ['Predicted NT type', 'nt_type', 'predicted_nt'], False)
    ids = attrs[idcol].tolist()
    lookup = dict(zip(ids, range(len(ids))))
    pre_parts, post_parts, count_parts = [], [], []
    invalid_rows = 0
    missing = set()
    headers = pd.read_csv(conn_path, nrows=0)
    precol = column(headers, columns['pre'], ['pre_root_id', 'pre_pt_root_id', 'Pre Root ID', 'source', 'pre_id'])
    postcol = column(headers, columns['post'], ['post_root_id', 'post_pt_root_id', 'Post Root ID', 'target', 'post_id'])
    countcol = column(headers, columns['count'], ['syn_count', 'synapse_count', 'Synapse Count', 'weight', 'synapses'])
    print(f'Columns: pre={precol}, post={postcol}, count={countcol}; NT={verified}/{predicted}')
    for chunk in pd.read_csv(conn_path, dtype=str, keep_default_na=False,
                             usecols=[precol, postcol, countcol], chunksize=250000):
        pre = checked_ids(chunk[precol]); post = checked_ids(chunk[postcol])
        counts = pd.to_numeric(chunk[countcol], errors='coerce').to_numpy(float)
        if not np.all(np.isfinite(counts)) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError('Connection weights must be nonnegative integer synapse counts.')
        for series in (pre, post):
            for value in series.unique():
                if value not in lookup:
                    missing.add(value)
                    lookup[value] = len(ids); ids.append(value)
        valid = counts > 0
        invalid_rows += int((~valid).sum())
        pre_parts.append(pre.map(lookup).to_numpy(np.int64)[valid])
        post_parts.append(post.map(lookup).to_numpy(np.int64)[valid])
        count_parts.append(counts[valid])
    if not count_parts:
        raise ValueError('Connections CSV is empty.')
    pre, post, counts = map(np.concatenate, (pre_parts, post_parts, count_parts))
    if len(counts) == 0:
        raise ValueError('No positive connections found.')
    graph_ids = set(np.concatenate([pre, post]).tolist())
    original_count = len(attrs)
    overlap = sum(i < original_count for i in graph_ids)
    if overlap == 0:
        raise ValueError('No connection IDs match attributes. Check dataset and release.')
    classes = np.full(len(ids), 'disabled', dtype='<U8')
    rules = {normalized(k): v for k, v in cfg['nt_class_map'].items()}
    unknown_labels = {}
    for i, (_, row) in enumerate(attrs.iterrows()):
        label = row[verified].strip() if verified else ''
        if not label and predicted:
            label = row[predicted].strip()
        kind = rules.get(normalized(label))
        if kind is None:
            unknown_labels[label or '(blank)'] = unknown_labels.get(label or '(blank)', 0) + 1
        else:
            classes[i] = kind
    for value, kind in cfg['neuron_class_overrides'].items():
        if value not in lookup:
            raise ValueError(f'Override ID absent: {value}')
        classes[lookup[value]] = kind
    if not set(classes).issubset({'exc', 'inh', 'disabled'}):
        raise ValueError('NT classes must be exc, inh, or disabled.')
    disabled_sources = int((classes[np.unique(pre)] == 'disabled').sum())
    if disabled_sources and settings['unmapped_transmitters'] == 'error':
        raise ValueError(f'{disabled_sources} presynaptic neurons have no supported synaptic class. '
                         f'Labels: {unknown_labels}. Add reviewed mappings or explicitly use '
                         'unmapped_transmitters="disable" to omit their outgoing effects.')
    if settings['unmapped_transmitters'] not in ('error', 'disable'):
        raise ValueError('unmapped_transmitters must be error or disable.')
    populations = cfg['populations']
    for name in ('input_left', 'input_right', 'output_left', 'output_right'):
        values = populations[name]
        if not values or any(not isinstance(v, str) for v in values):
            raise ValueError(f'Populate {name} with quoted BANC ID strings.')
        if len(set(values)) != len(values):
            raise ValueError(f'Duplicate IDs in {name}.')
    model = NeuralModel(ids, pre, post, counts, classes, cfg['neural'])
    for values in populations.values():
        model.indices(values)
    report = dict(mode='BANC ASSUMPTION-BASED MODEL', neurons=len(ids),
                  raw_positive_rows=len(counts), active_unique_edges=model.exc.nnz + model.inh.nnz,
                  graph_attribute_overlap=overlap / len(graph_ids),
                  missing_attribute_neurons=len(missing), zero_count_rows=invalid_rows,
                  disabled_sources=disabled_sources,
                  disabled_edge_rows=int((classes[pre] == 'disabled').sum()),
                  unrecognized_labels=unknown_labels)
    return model, populations, report
