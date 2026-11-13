"""Local recording index; does not change episode contents or training labels."""
import argparse
import hashlib
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def link(path, target):
    target = target.resolve()
    if path.is_symlink():
        if path.resolve() == target:
            return
        path.unlink()
    elif path.exists():
        if path.resolve() == target:
            return
        raise FileExistsError(path)
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=target.is_dir())


def register_episode(episode, root=ROOT):
    episode = Path(episode).resolve()
    if not (episode / 'trajectory.jsonl').exists():
        return
    home = root / 'recordings'
    group = home / ('episodes' if (episode / 'recording.json').exists() else 'legacy')
    group.mkdir(parents=True, exist_ok=True)
    if episode.parent != group.resolve():
        name = episode.name
        entry = group / name
        if (entry.exists() or entry.is_symlink()) and entry.resolve() != episode:
            name = episode.parent.name + '--' + name
            entry = group / name
        if (entry.exists() or entry.is_symlink()) and entry.resolve() != episode:
            raise FileExistsError(entry)
        link(entry, episode)
    refresh(root)


def refresh_seed_view(home, entries):
    """Maintain a browsing view; canonical recordings and export paths stay put."""
    view = home / 'by-seed'
    view.mkdir(exist_ok=True)
    previous = read_json(view / 'manifest.json')
    rows = []
    for item in entries:
        if item['group'] != 'episodes':
            continue
        seed = item['seed']
        group = f'seed-{seed:03d}' if type(seed) is int and seed >= 0 else 'seed-unknown'
        stamp = datetime.fromtimestamp(item['timestamp']).strftime('%Y%m%d-%H%M%S')
        suffix = hashlib.sha256(item['source'].encode()).hexdigest()[:8]
        relative = f'{group}/episode-{stamp}-{suffix}'
        alias = view / relative
        alias.parent.mkdir(exist_ok=True)
        target = Path(item['source'])
        if alias.is_symlink() and alias.resolve() != target.resolve():
            raise FileExistsError(alias)
        link(alias, target)
        item['seed_view_path'] = str(alias.relative_to(home))
        rows.append(dict(item, view_path=relative))
    desired = {r['view_path'] for r in rows}
    for old in previous.get('episodes', []):
        relative = Path(old['view_path'])
        if (relative.is_absolute() or len(relative.parts) != 2
                or not relative.parts[0].startswith('seed-') or '..' in relative.parts):
            continue
        alias = view / relative
        if (str(relative) not in desired and alias.is_symlink()
                and alias.resolve() == Path(old['source']).resolve()):
            alias.unlink()
            # Only remove empty generated seed directories, never user files.
            if not any(alias.parent.iterdir()):
                alias.parent.rmdir()
    payload = dict(kind='local_browsing_links_not_delivery_dataset',
                   episode_count=len(rows), seed_count=len({r['view_path'].split('/')[0] for r in rows}),
                   episodes=rows)
    (view / 'manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    (view / 'README.txt').write_text(
        '按种子查看原始回合\n\nseed-NNN/：同一种子的多次运行；episode-日期-时间-标识：独立一回合。\n'
        '这里只放快捷入口，不复制图片或移动原始数据；manifest.json记录原名称、路径和验收标签。\n'
        '不同日期可能使用不同场景/控制版本，具体看每回合report.json和source/。\n'
        '报告成功不等于训练合格；原有行为未通过标记保留。旧版实验仍在../legacy/。\n'
        '最终交付使用../../datasets/下经验证的LeRobot数据集，不单独打包此快捷入口目录。\n')


def refresh(root=ROOT):
    home = root / 'recordings'
    for name in ('episodes', 'legacy', 'replays'):
        (home / name).mkdir(parents=True, exist_ok=True)
    entries = []
    for group in ('episodes', 'legacy'):
        for path in (home / group).iterdir():
            if not (path / 'trajectory.jsonl').exists():
                continue
            report = read_json(path / 'report.json')
            config = read_json(path / 'run-config.json')
            quality = read_json(path / 'quality.json')
            behaviour = read_json(path / 'behaviour-review.json')
            stamp_path = path / 'run-config.json'
            if not stamp_path.exists():
                stamp_path = path / 'trajectory.jsonl'
            stamp = stamp_path.stat().st_mtime
            video = path / 'replay-trimmed.webm'
            if not video.exists():
                video = path / 'replay-trimmed.mp4'
            video_link = home / 'replays' / (path.name + video.suffix)
            if video.exists():
                link(video_link, video)
                # Keep one playback entry per episode; retain the original file.
                if video.suffix == '.webm':
                    old_link = home / 'replays' / (path.name + '.mp4')
                    if old_link.is_symlink() and old_link.resolve() == (path / 'replay-trimmed.mp4').resolve():
                        old_link.unlink()
            entries.append(dict(name=path.name, group=group, path=str(path.relative_to(home)),
                source=str(path.resolve()), timestamp=stamp,
                time=datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S'),
                seed=report.get('seed', config.get('seed', '—')),
                scene_version=report.get('scene_version', '未记录'),
                success=report.get('success'), completed=report.get('episode_completed'),
                frames=report.get('frames', quality.get('committed_frames', '—')),
                integrity=quality.get('integrity_passed'),
                behaviour_training_approved=behaviour.get('training_approved'),
                video=str(video_link.relative_to(home)) if video.exists() else None))
    entries.sort(key=lambda r: r['timestamp'], reverse=True)
    refresh_seed_view(home, entries)
    modern = [r for r in entries if r['group'] == 'episodes']
    if modern:
        link(home / 'latest', home / modern[0]['path'])
    elif (home / 'latest').is_symlink():
        (home / 'latest').unlink()
    successful = [r for r in modern if r['success'] is True and r['integrity'] is True]
    if successful:
        link(home / 'latest-success', home / successful[0]['path'])
    elif (home / 'latest-success').is_symlink():
        (home / 'latest-success').unlink()
    link(home / 'act-datasets', root / 'datasets')
    def anchor(url, label):
        return f'<a href="{quote(url, safe="/")}">{escape(str(label))}</a>'
    def table(items):
        body = []
        for item in items:
            state = '成功' if item['success'] is True else ('失败' if item['success'] is False else '未完成／无验收')
            integrity = {True: '通过', False: '未通过', None: '未检查'}[item['integrity']]
            path = item.get('seed_view_path', item['path'])
            views = [anchor(path + '/' + c + '/', label) for c, label in
                [('harvest_close', '近景'), ('wrist', '手腕'), ('top', '全景')]
                if (home / path / c).is_dir()]
            video = anchor(item['video'], '播放回放') if item['video'] else '未生成视频'
            result = anchor(path + '/report.json', state) if (home / path / 'report.json').exists() else state
            if item['behaviour_training_approved'] is False:
                result += '<br>' + anchor(path+'/behaviour-review.json','行为未通过 · 不用于训练')
            body.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in [
                escape(item['time']), escape(str(item['seed'])), escape(str(item['scene_version'])), result, integrity,
                escape(str(item['frames'])), video,
                anchor(path + '/', item['name']) + '<br>' + ' · '.join(views)]) + '</tr>')
        return '<table><thead><tr><th>运行时间</th><th>Seed</th><th>场景版本</th><th>记录结果</th><th>完整性</th><th>帧数</th><th>回放</th><th>原始数据</th></tr></thead><tbody>' + ''.join(body) + '</tbody></table>'
    legacy = [r for r in entries if r['group'] == 'legacy']
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>草莓采摘记录</title>
<style>body{font:16px system-ui,sans-serif;margin:32px;color:#213129;background:#f8faf7}h1{margin-bottom:8px}p{line-height:1.7}a{color:#176c47}table{width:100%;border-collapse:collapse;background:white;margin:18px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid #dfe7df}th{background:#eaf2e9}td:last-child{font-size:13px;overflow-wrap:anywhere}details{margin-top:32px}nav a{margin-right:24px}</style>
<h1>草莓采摘记录</h1><p>每个文件夹是一回合。成功示范目标：采一颗成熟草莓、送筐并返回。<br>
“成功”来自该回合原始报告；完整性通过不等于已完成训练验收。旧版实验与当前数据分开列出。视频是观看副本，原始帧和动作保留。</p>'''
    navigation = [('by-seed/', '按种子打开文件夹'), ('latest/', '最新回合'),
                  ('latest-success/', '最近成功回合'), ('replays/', '回放视频文件夹'),
                  ('act-datasets/', 'ACT 数据集'), ('../GUIDE.md', '操作与排错经验')]
    html += '<nav>' + ''.join(anchor(path, label) for path, label in navigation
                             if (home / path).exists()) + '</nav>'
    if not entries:
        html += '<p>当前没有采集回合。新回合完成记录并注册后，将自动按种子归类。</p>'
    deleted = read_json(home / 'deleted-failures.json')
    if deleted:
        count = sum(e.get('status') == 'deleted' for e in deleted.get('entries', []))
        html += f'<p>已按要求清理 {count} 个失败或未完成回合；此列表不能用于重算成功率。<a href="deleted-failures.json">查看清理清单与失败摘要</a>。新失败仍会正常保存。</p>'
    html += f'<h2>按种子查看 · {len(modern)} 个当前格式回合</h2><p>同一种子可有多次运行及不同控制版本，均独立保留；分组不代表训练验收。</p>'
    seed_groups = sorted({r['seed_view_path'].split('/')[1] for r in modern})
    for group in seed_groups:
        items = [r for r in modern if r['seed_view_path'].split('/')[1] == group]
        html += f'<details open><summary>{anchor("by-seed/" + group + "/", group)} · {len(items)} 回合</summary>' + table(items) + '</details>'
    html += f'<details><summary>旧版实验记录 · {len(legacy)} 回合（不作为当前成功率）</summary>' + table(legacy) + '</details></html>'
    (home / 'index.html').write_text(html)
    (home / 'catalog.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2) + '\n')
    (home / 'README.txt').write_text('草莓采摘记录\n\n双击 index.html：按时间查看每回合结果、各路图片和已有回放。\n'
        '项目 GUIDE.md：控制经验、运行命令、验收标准和排错方法。\n'
        'by-seed/：日常浏览入口，一个种子一个文件夹，内含各次回合的快捷入口。\n'
        'episodes/：原始记录与兼容入口，保留原路径供回放、导出及历史报告使用。\n'
        'replays/：已生成的回放快捷入口，优先 WebM/VP8，不复制视频。\n'
        'latest/：最新回合；latest-success/：最近报告成功且完整性通过的回合。\n'
        'legacy/：旧版实验的快捷入口，不作为当前稳定性验证。\n'
        'act-datasets/：已导出的 ACT 格式样例，尚非正式交付数据集。\n\n'
        '每回合内：harvest_close、wrist 是新回合两路逐帧图片，旧回合还含 top；trajectory.jsonl 是逐帧状态和动作；\n'
        'report.json 是结果；quality.json 是完整性检查。文件夹内部路径保持原格式。\n'
        '新回合默认保存到 episodes；指定 --out 的回合会自动建立入口。回放视频需单独生成。\n'
        '若存在 deleted-failures.json，表示历史失败已清理，清单保留简要原因；不能按剩余记录重算成功率。\n')
    return entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path, nargs='?')
    args = parser.parse_args()
    if args.episode:
        register_episode(args.episode)
    else:
        refresh()
