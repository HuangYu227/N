"""Plot measured, accepted results only; no illustrative synthetic success curves."""
import argparse
import json
from pathlib import Path

from memory_math import MODES, check_clean_pair


def load_results(paths):
    manifests, records = [], {}
    for path in paths:
        path = Path(path)
        acceptance = json.loads((path/'acceptance.json').read_text())
        if acceptance.get('status') != 'passed':
            raise ValueError('Diagnostic did not pass runtime acceptance')
        manifest = json.loads((path/'manifest.json').read_text())
        manifests.append(manifest)
        for mode in acceptance['modes']:
            record = json.loads((path/(mode+'.json')).read_text())
            if mode in records or record.get('status') != 'completed':
                raise ValueError('Duplicate or unfinished mode')
            records[mode] = record
    if set(records) != set(MODES):
        raise ValueError('All four completed modes are required')
    keys = ('checkpoint_sha256', 'inputs', 'source_hashes', 'seed', 'frames',
            'prefix_probes', 'selected_chunks', 'requested_sigmas', 'camera_cache_chunks', 'precision')
    for manifest in manifests[1:]:
        if any(manifest[key] != manifests[0][key] for key in keys):
            raise ValueError('Checkpoint, code, inputs or protocols differ between GPU jobs')
    check_clean_pair(records['full'], records['no-local'])
    check_clean_pair(records['no-persistent'], records['no-ttt'])
    if any(record['prefix_hashes'] != records['full']['prefix_hashes'] for record in records.values()):
        raise ValueError('Observed prefixes differ between ablations')
    velocity_keys = lambda record: [(row['chunk'], row['production_index'], row['sigma'])
                                     for row in record['velocity']]
    if any(velocity_keys(record) != velocity_keys(records['full']) for record in records.values()):
        raise ValueError('Velocity measurements are not paired')
    return manifests[0], records


def plot(paths, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    manifest, records = load_results(paths)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = {'full': '#2667a8', 'no-local': '#e09832', 'no-persistent': '#34836b', 'no-ttt': '#a5527c'}
    labels = {'full': 'Both TTT branches', 'no-local': 'Clean TTT only',
              'no-persistent': 'Local TTT only', 'no-ttt': 'Correct only'}
    anchors = sorted({row['anchor'] for row in records['full']['memory']})
    files = []

    def save(fig, name):
        fig.savefig(output/(name+'.png'), dpi=200, bbox_inches='tight')
        fig.savefig(output/(name+'.pdf'), bbox_inches='tight')
        files.append(name+'.png')
        plt.close(fig)

    for metric, title, ylabel in (
        ('old_key_binding_nmse', 'Frozen observed-prefix K-to-V binding', 'Normalized feature MSE'),
        ('old_query_response_drift', 'Frozen observed-prefix Q response change', 'Normalized response change')):
        fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharex=True)
        for ax, anchor in zip(axes, anchors):
            for mode in MODES:
                rows = [r for r in records[mode]['memory'] if r['anchor'] == anchor]
                ax.plot([r['chunk'] for r in rows], [r[metric] for r in rows],
                        label=labels[mode], color=colors[mode], linewidth=1.7)
            ax.set_title(f'Anchor {anchor}')
            ax.set_xlabel('GT clean commits')
            ax.grid(alpha=.2)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8)
        fig.suptitle(title+' (known value rotation compensated; proxy, not video quality)', y=1.08)
        save(fig, metric)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    sigmas = records['full']['production_sigmas']
    for ax, sigma in zip(axes, sigmas):
        for mode in MODES:
            rows = [r for r in records[mode]['velocity'] if r['sigma'] == sigma]
            ax.plot([r['chunk'] for r in rows], [r['mse'] for r in rows], marker='o',
                    color=colors[mode], label=labels[mode], markersize=3)
        ax.set_title(f'Production sigma={sigma:.3f}')
        ax.set_xlabel('Probe chunk'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Conditional velocity MSE')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Paired GT-history denoising probes (training example; not autoregressive quality)', y=1.04)
    save(fig, 'velocity_mse')

    fig, ax = plt.subplots(figsize=(6, 4))
    for anchor in anchors:
        rows = [r for r in records['full']['memory'] if r['anchor'] == anchor and r.get('inner')]
        x = [100*(r['inner']['loss_after']/max(r['inner']['loss_before'], 1e-30)-1) for r in rows]
        ax.scatter(x, [r['ttt_old_binding_delta'] for r in rows], label=f'Anchor {anchor}', s=13, alpha=.65)
    ax.axhline(0, color='black', linewidth=.7); ax.axvline(0, color='black', linewidth=.7)
    ax.set_xlabel('Current inner-objective change (%)')
    ax.set_ylabel('Old binding NMSE: TTT + Correct minus Correct')
    ax.set_title('Same features and pre-update S; correlated descriptive points')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig, 'current_fit_vs_old_binding')

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    for ax, anchor in zip(axes, anchors):
        rows = [r for r in records['full']['memory'] if r['anchor'] == anchor]
        values = np.asarray([r['per_token_key_nmse'] for r in rows]).T
        im = ax.imshow(values, aspect='auto', origin='lower', cmap='viridis',
                       extent=[0, max(r['chunk'] for r in rows), 0, values.shape[0]])
        ax.set_title(f'Anchor {anchor}'); ax.set_xlabel('GT clean commits')
        fig.colorbar(im, ax=ax, fraction=.05)
    axes[0].set_ylabel('Sampled observed-prefix token')
    fig.suptitle('Old binding NMSE by frozen probe token (not attention weights)', y=1.05)
    save(fig, 'binding_heatmap')
    summary = dict(step=manifest['step'], case_id=manifest['case_id'], checkpoint_sha256=manifest['checkpoint_sha256'],
                   scope=manifest['protocol'], files=files, paired_velocity_deltas={})
    full = records['full']['velocity']
    for mode in MODES[1:]:
        summary['paired_velocity_deltas'][mode+' minus full'] = [
            dict(chunk=a['chunk'], sigma=a['sigma'], mse_delta=b['mse']-a['mse'])
            for a, b in zip(full, records[mode]['velocity'])]
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.results, args.output), indent=2))
