import sys


def progressbar(total, i, bar_length=50, prefix='', suffix=''):
    """progressbar
    """
    bar_graph = '█'
    if i % max((total // 100), 1) == 0:
        dot_num = int((i + 1) / total * bar_length)
        dot = bar_graph * dot_num
        empty = '.' * (bar_length - dot_num)
        sys.stdout.write(f'\r {prefix} [{dot}{empty}] {i / total * 100:3.2f}% {suffix}')
    if i == total:
        sys.stdout.write(f'\r {prefix} [{bar_graph * bar_length}] {100:3.2f}% {suffix}')
        sys.stdout.write('Done ')
