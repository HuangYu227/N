"""Reject incomparable/incomplete GPU outputs before drawing any research figure."""
import json

import pytest

from plot_diagnostic import load_results


def write_pair(root):
    common = dict(checkpoint_sha256='weight', inputs={'clip': 'same'}, source_hashes={'code': 'same'},
                  seed=3407, frames=121, prefix_probes=64, selected_chunks=[2, 40],
                  requested_sigmas=[.2, .5, .8], camera_cache_chunks=2, precision='fp32')
    paths = [root/'gpu6', root/'gpu7']
    for path, modes in zip(paths, (['full', 'no-local'], ['no-persistent', 'no-ttt'])):
        path.mkdir()
        (path/'manifest.json').write_text(json.dumps(common))
        (path/'acceptance.json').write_text(json.dumps(dict(status='passed', modes=modes)))
        for mode in modes:
            record = dict(status='completed', clean_state_hashes=['state'], prefix_hashes=['prefix'],
                          velocity=[dict(chunk=2, production_index=1, sigma=.2, mse=.5)])
            (path/(mode+'.json')).write_text(json.dumps(record))
    return paths


def change(path, field, value):
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))


def test_only_accepted_paired_measurements_can_be_plotted(tmp_path):
    paths = write_pair(tmp_path)
    _, records = load_results(paths)
    assert len(records) == 4
    change(paths[1]/'manifest.json', 'checkpoint_sha256', 'other-weight')
    with pytest.raises(ValueError, match='protocols differ'):
        load_results(paths)


@pytest.mark.parametrize('field,value', [('status', 'running'), ('prefix_hashes', ['other-prefix'])])
def test_incomplete_or_different_prefix_fails(tmp_path, field, value):
    paths = write_pair(tmp_path)
    change(paths[1]/'no-ttt.json', field, value)
    with pytest.raises((ValueError, AssertionError)):
        load_results(paths)


def test_missing_mode_is_not_silently_omitted(tmp_path):
    paths = write_pair(tmp_path)
    change(paths[0]/'acceptance.json', 'modes', ['full'])
    with pytest.raises(ValueError, match='All four'):
        load_results(paths)
