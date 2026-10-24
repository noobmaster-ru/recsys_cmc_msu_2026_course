"""Data-only feasibility audit. No model imports, training, or held-out label analysis."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/Video_Games.csv.gz'
QUERY = 1546300800
END = 1577836800
COLUMNS = ['user_id', 'item_id', 'rating', 'timestamp']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def gzip_csv(path, frame):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding='utf-8', newline='') as out:
                frame.to_csv(out, index=False)


def core(frame, minimum=5):
    rounds = []
    while len(frame):
        uc = frame.groupby('user_id').user_id.transform('size')
        ic = frame.groupby('item_id').item_id.transform('size')
        keep = (uc >= minimum) & (ic >= minimum)
        if keep.all():
            break
        rounds.append(int((~keep).sum()))
        frame = frame.loc[keep].copy()
    return frame, rounds


def select_fit(frame, *, max_items=20000, max_events=1500000):
    """Selection has no test argument; reject any row reaching the query boundary."""
    if frame.empty or frame.timestamp.max() >= QUERY:
        raise ValueError('Selection requires nonempty fit strictly before 2019')
    if frame.duplicated(['user_id', 'item_id']).any():
        raise ValueError('Deduplicate fit pairs before selecting')
    selected, first_rounds = core(frame)
    before = len(selected)
    counts = selected.groupby('item_id').size().rename('events').reset_index()
    ranked = counts.sort_values(['events', 'item_id'], ascending=[False, True], kind='stable')
    selected = selected.loc[selected.item_id.isin(ranked.item_id.iloc[:max_items])].copy()
    selected, cap_rounds = core(selected)
    budget_used = len(selected) > max_events
    if budget_used:
        sizes = selected.groupby('user_id').size().rename('events').reset_index()
        sizes['order'] = sizes.user_id.map(lambda u: hashlib.sha256(('external2019:42:' + u).encode()).hexdigest())
        sizes = sizes.sort_values(['order', 'user_id'], kind='stable')
        keep_users = sizes.loc[sizes.events.cumsum() <= max_events, 'user_id']
        selected = selected.loc[selected.user_id.isin(keep_users)].copy()
    selected, budget_rounds = core(selected)
    return selected.sort_values(['user_id', 'item_id'], kind='stable').reset_index(drop=True), {
        'initial_fit_core_events': before, 'initial_core_removed_per_round': first_rounds,
        'post_item_cap_removed_per_round': cap_rounds, 'user_budget_applied': budget_used,
        'post_user_budget_removed_per_round': budget_rounds,
    }


def main():
    reports = ROOT / 'reports'
    local = ROOT / 'data'
    reports.mkdir(exist_ok=True)
    local.mkdir(exist_ok=True)
    plan_path = reports / 'selection_plan.json'
    if plan_path.exists():
        raise RuntimeError('Existing plan: refusing to overwrite or adapt completed selection')
    repository = ROOT.parents[1]
    old_manifest = repository / 'homework_1/reports/submission_manifest.json'
    protected = json.loads(old_manifest.read_text())
    for rel in ['experiments.ipynb', 'README.md', 'reports/submission_manifest.json']:
        protected[rel] = sha256(repository / 'homework_1' / rel)
    # Verify the original solution before any audit work; store its exact hashes.
    for rel, digest in protected.items():
        if sha256(repository / 'homework_1' / rel) != digest:
            raise ValueError('Original v2 evidence changed: ' + rel)
    plan = {
        'source': URL, 'source_kind': 'unsplit 0-core, not full-period 5-core',
        'fit': 'timestamp < 2019-01-01T00:00:00Z',
        'test': '2019-01-01T00:00:00Z <= timestamp < 2020-01-01T00:00:00Z',
        'timestamp_conversion': 'source milliseconds; filter before conversion; floor to Unix seconds',
        'fit_duplicates': 'earliest millisecond timestamp, ties original file row order',
        'fit_selection': ['iterative 5-core on fit only',
                          'at most 20000 items by fit event count descending, ties parent_asin ascending',
                          'repeat fit-only 5-core',
                          'if above 1500000 fit events: whole-user prefix ordered by SHA256(external2019:42:ID)',
                          'repeat fit-only 5-core'],
        'test_selection': 'apply frozen selected fit user/item IDs; do not tune filters using test',
        'test_label_policy': 'copy opaque rating strings to sealed CSV only; never parse or summarize test ratings',
        'test_feasibility_only': 'all-rating event counts, known IDs, seen masks, candidate availability; no relevance-conditioned cohort',
        'future_after_2019': 'count timestamps only, discard rows',
        'max_fit_events': 1500000, 'max_items': 20000, 'minimum_fit_degree': 5,
        'models_trained': False, 'recommendation_metrics_computed': False,
        'script_sha256': sha256(Path(__file__)), 'protected_v2_files': protected,
    }
    write_json(plan_path, plan)  # Immutable plan BEFORE download/inspection.
    source = local / 'Video_Games_0core_2023.csv.gz'
    if source.exists():
        raise RuntimeError('Unexpected preexisting source; inspect provenance before use')
    print('Plan frozen; downloading official 0-core...', flush=True)
    temp = source.with_suffix('.gz.part')
    with urllib.request.urlopen(URL, timeout=60) as response:
        headers = {k: response.headers.get(k) for k in ['Content-Length', 'Last-Modified', 'ETag']}
        with temp.open('wb') as out:
            total = 0
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                total += len(chunk)
                if total // (20 * 1024 * 1024) != (total-len(chunk)) // (20 * 1024 * 1024):
                    print(f'Downloaded {total // 1024**2} MiB', flush=True)
    if headers['Content-Length'] and total != int(headers['Content-Length']):
        raise ValueError('Incomplete source')
    temp.replace(source)
    write_json(reports / 'source.json', {'url': URL, 'bytes': total, 'sha256': sha256(source), 'headers': headers})
    fit_rows = []
    counts = {'fit_before_2019': 0, 'test_2019': 0, 'after_2019': 0}
    holdout = local / 'test_2019_all.csv.gz'
    print('Reading IDs/timestamps; sealing test ratings without analysis...', flush=True)
    with gzip.open(source, 'rt', encoding='utf-8', newline='') as inp, gzip.open(holdout, 'wt', encoding='utf-8', newline='') as out:
        reader = csv.DictReader(inp)
        if reader.fieldnames != ['user_id', 'parent_asin', 'rating', 'timestamp']:
            raise ValueError('Unexpected source schema')
        writer = csv.writer(out)
        writer.writerow(COLUMNS)
        for index, row in enumerate(reader):
            if not row['user_id'] or not row['parent_asin']:
                raise ValueError('Empty identifier')
            millis = int(row['timestamp'])
            if millis <= 0:
                raise ValueError('Invalid timestamp')
            if millis < QUERY * 1000:
                rating = float(row['rating'])  # Parse ONLY fit labels.
                if rating not in (1, 2, 3, 4, 5):
                    raise ValueError('Invalid fit rating')
                fit_rows.append((row['user_id'], row['parent_asin'], int(rating), millis // 1000, millis, index))
                counts['fit_before_2019'] += 1
            elif millis < END * 1000:
                # Rating stays an opaque string and is not used in feasibility decisions.
                writer.writerow([row['user_id'], row['parent_asin'], row['rating'], millis // 1000])
                counts['test_2019'] += 1
            else:
                counts['after_2019'] += 1
            if (index + 1) % 1000000 == 0:
                print(f'Read {index + 1} source rows', flush=True)
    frame = pd.DataFrame.from_records(fit_rows, columns=COLUMNS + ['_millis', '_source_order'])
    del fit_rows
    frame['rating'] = frame.rating.astype('int8')
    frame = frame.sort_values(['_millis', '_source_order'], kind='stable')
    duplicate_count = int(frame.duplicated(['user_id', 'item_id']).sum())
    frame = frame.drop_duplicates(['user_id', 'item_id'], keep='first')[COLUMNS].copy()
    before_stats = {'events': len(frame), 'users': frame.user_id.nunique(), 'items': frame.item_id.nunique()}
    selected, selection = select_fit(frame)
    del frame
    if selected.empty:
        raise ValueError('No fit core survived')
    fit_path = local / 'fit_before_2019.csv.gz'
    gzip_csv(fit_path, selected)
    users, items = set(selected.user_id), set(selected.item_id)
    # This path deliberately has no rating column.
    test_ids = pd.read_csv(holdout, usecols=['user_id', 'item_id', 'timestamp'], dtype={'user_id': str, 'item_id': str, 'timestamp': 'int64'})
    known_u = test_ids.user_id.isin(users)
    known_i = test_ids.item_id.isin(items)
    warm = test_ids.loc[known_u & known_i].copy()
    known_pairs = pd.MultiIndex.from_frame(selected[['user_id', 'item_id']])
    warm_pairs = pd.MultiIndex.from_frame(warm[['user_id', 'item_id']])
    unseen = ~warm_pairs.isin(known_pairs)
    eligible_events = warm.loc[unseen]
    history = selected.groupby('user_id').size()
    candidate_users = set(history.index[(len(items) - history) >= 10])
    available = eligible_events.loc[eligible_events.user_id.isin(candidate_users)]
    selected_holdout = local / 'test_2019_selected.csv.gz'
    # Recopy selected rows without interpreting their sealed labels.
    with gzip.open(holdout, 'rt', encoding='utf-8', newline='') as inp, gzip.open(selected_holdout, 'wt', encoding='utf-8', newline='') as out:
        reader = csv.DictReader(inp)
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        for row in reader:
            if row['user_id'] in users and row['item_id'] in items:
                writer.writerow(row)
    stats = {
        'events': len(selected), 'users': len(users), 'items': len(items),
        'minimum_user_events': int(history.min()),
        'minimum_item_events': int(selected.groupby('item_id').size().min()),
        'positive_fit_events_rating_ge_4': int((selected.rating >= 4).sum()),
        'first_timestamp': int(selected.timestamp.min()), 'last_timestamp': int(selected.timestamp.max()),
        'query_timestamp': QUERY, 'all_sources_strictly_before_query': bool((selected.timestamp < QUERY).all()),
        'ease_single_float64_matrix_GiB': len(items) ** 2 * 8 / 1024 ** 3,
    }
    # Confirm every previously tracked v2 source/report/config and main presentation is unchanged.
    for rel, digest in protected.items():
        if sha256(repository / 'homework_1' / rel) != digest:
            raise ValueError('Original v2 evidence modified: ' + rel)
    result = {
        'source': json.loads((reports / 'source.json').read_text()),
        'selection_plan_sha256': sha256(plan_path), 'source_rows_by_period': counts,
        'raw_fit_unique_pairs': before_stats, 'duplicate_fit_rows_removed': duplicate_count,
        'selection': selection, 'selected_fit': stats,
        'test_2019_structure_only': {
            'all_category_events': len(test_ids), 'unique_users_all_ratings': int(test_ids.user_id.nunique()),
            'unique_items_all_ratings': int(test_ids.item_id.nunique()),
            'events_with_known_user': int(known_u.sum()), 'events_with_known_item': int(known_i.sum()),
            'events_with_both_known': len(warm), 'users_with_both_known_events': int(warm.user_id.nunique()),
            'seen_fit_pairs_in_warm_test': int((~unseen).sum()),
            'available_unseen_events_all_ratings': len(available),
            'users_with_at_least_one_available_event_and_10_candidates': int(available.user_id.nunique()),
            'relevance_conditioned_user_count': None,
            'positive_test_event_count': None,
            'note': 'These counts include all ratings; eligible ranking users may be fewer. No test label analysis or model metrics.',
        },
        'assignment_size_checks': {
            'fit_at_least_100000_events': len(selected) >= 100000,
            'fit_at_least_1000_users': len(users) >= 1000,
            'fit_at_least_1000_items': len(items) >= 1000,
            'selected_fit_plus_selected_test_at_most_2000000_events': len(selected) + len(warm) <= 2000000,
        },
        'artifacts': {str(p.relative_to(ROOT)): sha256(p) for p in [source, fit_path, holdout, selected_holdout]},
        'models_trained': False, 'test_recommendation_metrics_computed': False,
        'test_labels_used_for_selection': False, 'test_labels_analyzed': False,
        'protected_v2_files_verified_unchanged': len(protected),
    }
    write_json(reports / 'feasibility.json', result)
    print(json.dumps({k: result[k] for k in ['selected_fit', 'test_2019_structure_only', 'assignment_size_checks']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
