#!/usr/bin/env python3
import datetime
import os
import pprint
import shutil
import sys
import tabulate
import termcolor
import types

from .utils import get_env_flag, get_repo_dir, to_bool


# Logging functions for different alert levels
def log_error(x, **kwargs):
    """ Log an error message """
    print_log(x, level='error', **kwargs)

def log_warning(x, **kwargs):
    """ Log a warning message """
    print_log(x, level='warning', **kwargs)

def log_success(x, **kwargs):
    """ Log a success message """
    print_log(x, level='success', **kwargs)

def log_info(x, **kwargs):
    """ Log an info message """
    print_log(x, level='info', **kwargs)

def log_verbose(x, **kwargs):
    """ Log a verbose/debug message """
    print_log(x, level='verbose', **kwargs)

def log_debug(x, **kwargs):
    """ Log a verbose/debug message """
    print_log(x, level='debug', **kwargs)


# Generator for setting templates, prefix text, colors/dates, ect.
def LogLevel(**kwargs):
    """ Return a logging level type with the given defaults """
    return types.SimpleNamespace(**kwargs)


#
# Global logging configuration
#
#   * see `log_config()` for a runtime API for modifying this
#   * configurable from CLI with --log-dir, --log-level, ect
#   * enable/disable terminal color codes with --log-colors=off
#
LogConfig = types.SimpleNamespace(

    # The active logging level (messages below this level are suppressed)
    level = 'debug' if get_env_flag('DEBUG') else \
            'verbose' if get_env_flag('VERBOSE') else 'info',

    # Definitions of the log levels, increasing incrementally by priority
    levels = {
        'debug':    LogLevel(color='light_grey'),
        'verbose':  LogLevel(color='light_grey'),
        'info':     LogLevel(),
        'status':   LogLevel(color='blue'),
        'success':  LogLevel(color='green'), # prefix=' ✅ '
        'warning':  LogLevel(color='yellow'), # prefix=' ⚠️ '
        'error':    LogLevel(color='red'), # prefix=' ❌ '
    },

    # Default message format template (todo add $DATE)
    format = "[${TIME}]${PREFIX} ${MESSAGE} ${POSTFIX}",

    # Default prefix/suffix text (if not overriden by level)
    prefix = '',
    postfix = '',

    # Optional separator appended after substitutions
    separator = '',
    indent = 0,

    # Colors & terminal statusbar enabled or not
    colors = 'ANSI_COLORS_DISABLED' not in os.environ,
    status = None,

    # The default logging location is `jetson-containers/logs/<timestamp>`
    dir = os.path.join(
        get_repo_dir(), 'logs',
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ),
)


def log_config(key: str=None, dir: str=None, level: str=None,
               verbose: bool=None, debug: bool=None,
               colors: bool=None, status: bool=None, **kwargs):
    """
    Configures different settings of the logging system, including the output
    save directory, active verbosity level, terminal colors, ect.
    """
    dir = kwargs.get('log_dir', kwargs.get('logs', dir))
    level = kwargs.get('log_level', level)
    colors = kwargs.get('log_colors', colors)
    status = kwargs.get('log_status', status)

    falsy = ['off', 'false', '0', 'no', 'disabled', 'none']
    truthy = ['on', 'true', '1', 'yes', 'enabled']

    if verbose:
        level = 'verbose'

    if debug:
        level = 'debug'

    if level:
        if level not in LogConfig.levels:
            raise ValueError(f"log_config() got unexpected value '{level}' for level parameter (valid levels are {LogConfig.levels})")
        LogConfig.level = level

    if dir:
        LogConfig.dir = dir

    if colors is not None: # disable terminal colors
        LogConfig.colors = colors

    if status == False: # disable terminal status bar
        LogConfig.status = False # this gets set to true once used

    if key:
        return LogConfig.get(key)
    else:
        return LogConfig


def print_log(text: str='', level: str='info', color: str=None, attrs: list=[]):
    """
    Prints a logging message based on the currently defined level.
    """
    if level not in LogConfig.levels:
        log_warning(f"Unrecognized logging level '{level}' (valid levels are {LogConfig.levels})\n   {text}")
        return

    keys = list(LogConfig.levels.keys())

    if keys.index(level) < keys.index(LogConfig.level):
        return

    def substitution(txt, **kwargs):
        for k,v in kwargs.items():
            var_a = '$' + k.upper()
            var_b = '${' + k.upper() + '}'
            txt = txt.replace(var_a, var_b).replace(var_b, v + LogConfig.separator)
        return txt

    now = datetime.datetime.now()

    config = LogConfig.levels[level]
    indent = getattr(config, 'indent', LogConfig.indent)

    if indent:
        indent = ' ' * indent
        text = indent + f"\n{indent}".join(text.split('\n'))

    format = substitution(
        getattr(config, 'format', LogConfig.format),
        prefix=getattr(config, 'prefix', LogConfig.prefix),
        postfix=getattr(config, 'postfix', LogConfig.postfix),
        level=level.upper(),
        message=text,
        line_sep='\n' + '#' * 80 + '\n',
        datetime=now.strftime('%-m/%-d/%-y %H:%M:%S'),
        date=now.strftime('%y%m%d'),
        time=now.strftime('%H:%M:%S'),
    )

    if not color:
        color = getattr(config, 'color', None)

    cprint(format, color, attrs=attrs)


def pprint_debug(*args, **kwargs):
    """
    Debug print function that only prints when VERBOSE or DEBUG environment variable is set
    TODO change this to use python logging APIs or move to logging.py
    """
    if LogConfig.level == 'debug':
        pprint.pprint(*args, **kwargs)


def cprint(text, color=None, on_color=None, attrs=[], **kwargs):
    """
    Print string to terminal in the specified color.  The recognized colors are found below.
    """
    kwargs.setdefault('flush', True)
    print(colorize(text, color, on_color, attrs), **kwargs)


def colorize(text, color=None, on_color=None, attrs=[]):
    """
    Apply ANSI terminal color codes - supports some inline tags like `<b>Bold</b>`
    """
    if not text:
        return None

    if 'ANSI_COLORS_DISABLED' in os.environ or not LogConfig.colors:
        return text.replace('<b>', '').replace('</b>', '')

    if attrs and isinstance(attrs, str):
        attrs = [attrs]

    # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797#colors--graphics-mode
    text = text.replace('<b>', f'\033[1m').replace('</b>', f'\033[22m')

    if color or on_color or attrs:
        text = termcolor.colored(text, color=color, on_color=on_color, attrs=attrs)

    return text


def get_log_dir(subdir: str=None, create: bool=True):
    """
    Return the path to the logging directory (creating it if needed)
    The subdir used is typically one of:  build, test, run
    """
    path = LogConfig.dir

    if subdir:
        path = os.path.join(path, subdir)

    if create:
        os.makedirs(path, exist_ok=True)

    return path


def log_status(text='', prefix='', done=False, **kwargs):
    """
    Log to the status bar at the bottom of terminal.

    This changes the scrolling region with DECSTBM/DECOM:
      https://vt100.net/docs/vt102-ug/chapter5.html

    You need to call this with `done=True` the last time
    using it in order for the terminal to properly be reset.
    Using an exception handler to make sure is probably needed.
    """
    terminal = shutil.get_terminal_size(fallback=(80, 24))
    termcode = f'\0337\033[?6l\033[{terminal.lines};1H\033[2K'

    if text:
        log_info(text)

        if LogConfig.status != False and LogConfig.colors:
            kwargs.setdefault('color', 'green')
            kwargs.setdefault('attrs', 'reverse')

            if LogConfig.status != True:
                print(f'\033[1;{terminal.lines-1}r\033[?6l\033[2J\033[H', end='', flush=True)
                LogConfig.status = True

            text = prefix + text
            text += ' ' * (terminal.columns - len(text))
            text = colorize(text, color='green', attrs='reverse')

            print(f'{termcode}{text}\0338', end='', flush=True)

    if done and LogConfig.status:
        print(f'{termcode}\033[r\033[0m\033[{terminal.lines};1H')
        LogConfig.status = None


def log_versions(keys=None, columns=4, color='green', tablefmt='simple_grid'):
    """
    Prints a table of platform environment variables.
    """
    from jetson_containers import l4t_version

    if not keys:
        keys = [
            'L4T_VERSION', 'JETPACK_VERSION',
            'CUDA_VERSION', 'PYTHON_VERSION',
            'SYSTEM_ARCH', 'LSB_RELEASE',
        ]

    rows = []

    for key in keys:
        text = [key, f"{getattr(l4t_version, key)}"]
        if not rows or len(rows[-1]) >= columns:
            rows.append(text)
        else:
            rows[-1].extend(text)

    log_table(rows, merge_columns=True, attrs='reverse')


def log_block(*text, color='green', attrs=['reverse'], tablefmt='simple_grid', **kwargs):
    """
    Log a block of text inside a simple grid pattern (like a table, just one cell)
    """
    if not text:
        return None

    head = text[0]
    text = text[1:] if len(text) > 1 else None

    head = colorize(head, None)
    output = tabulate.tabulate([[head]], tablefmt=tablefmt, **kwargs)

    if not kwargs.get('visible', True):
        return output

    output = '\n'.join([
        colorize(x, color, attrs=attrs)
        for x in output.split('\n')
    ])

    print(f"\n{output}\n")

    if not text:
        return output

    for x in text:
        print(f"{colorize(x)}\n")

    return output


def log_table( rows, header=None, footer=None,
               filter=(), color='green', on_color=None,
               min_widths=[], max_widths=[30,55],
               wrap_rows=None, merge_columns=False,
               attrs=None, visible=True, **kwargs ):
    """
    Print a key-based table from a list[list] of rows/columns, or a 2-column dict
    where the keys are column 1, and the values are column 2.  These can be wrapped
    and merged, or recursively nested like a tree of dicts.

    Header is a list of columns or rows that are inserted at the top.
    Footer is a list of columns or rows that are added to the end.

    This uses tabulate for layout, and the kwargs are passed to it:
      https://github.com/astanin/python-tabulate

    Color names and text attributes are from termcolor library:
      https://github.com/termcolor/termcolor#text-properties
    """
    if min_widths is None:
        min_widths = []

    if max_widths is None:
        max_widths = []

    kwargs.setdefault('numalign', 'center')         # set alignment kwargs to 'left', 'right', 'center'
    kwargs.setdefault('tablefmt', 'simple_outline') # set 'tablefmt' kwarg to change style

    if isinstance(rows, dict):
        rows = flatten_rows(rows, filter=filter)

    for row in rows:
        for c, col in enumerate(row):
            col = str(col)
            if c < len(min_widths) and len(col) < min_widths[c]:
                col = format_str(col, min_widths[c], pad=True)
            if c < len(max_widths) and len(col) > max_widths[c]:
                col = col[:max_widths[c]]
            row[c] = col

    if header:
        if not isinstance(header[0], list):
            header = [header]
        rows = header + rows

    if footer:
        if not isinstance(footer[0], list):
            footer = [footer]
        rows = rows + footer

    if wrap_rows and len(rows) > wrap_rows:
        for i in range(wrap_rows, len(rows)):
            rows[i % wrap_rows].extend(rows[i])
        rows = rows[:wrap_rows]

    if merge_columns:
        if merge_columns == True:
            merge_columns = 2
        new_columns = int(len(rows[0]) / merge_columns)
        min_widths = [0] * new_columns
        for r, row in enumerate(rows):
            for nc in range(new_columns):
                if nc*merge_columns < len(row):
                    min_widths[nc] = max(min_widths[nc], len(row[nc*merge_columns]))
        for r, row in enumerate(rows):
            new_row = []
            for nc in range(new_columns):
                if nc*merge_columns >= len(row):
                    continue

                new_col = format_str(
                    row[nc*merge_columns],
                    length=min_widths[nc],
                    pad=True
                )
                for x in range(1,merge_columns):
                    new_col += '  ' + row[nc*merge_columns+x]
                new_row.append(new_col)
            rows[r] = new_row

    table = tabulate.tabulate(rows, **kwargs)

    '''
    if color:
        table = '\n'.join([
            colorize(x, color, attrs=attrs)
            for x in table.split('\n')
        ])
    '''

    if visible:
        cprint(table, color=color, on_color=on_color, attrs=attrs)

    return table


def flatten_rows(seq, filter=()):
    """
    Recursively convert a tree of list/dict/tuple objects to a flat list.
    This is so they can be printed in a simple two-column table.
    """
    def flatten(seq, indent='', prefix='', out=[]):
        iter = range(len(seq)) if isinstance(seq,(list,tuple)) else seq
        for key in iter:
            val = seq[key]
            if filter:
                val = filter(seq,key,val)
                if isinstance(val, tuple) and len(val) == 2:
                    key, val = val
            if not val:
                continue
            if isinstance(seq,dict) and isinstance(val,list):
                flatten(val, indent, f'{key} ', out)
            elif isinstance(val, (tuple,list,dict,map)):
                out.append([indent + prefix + str(key), ''])
                flatten(val, indent + (' ├ ' if len(val) > 1 else ''), out=out)  # ┣
            else:
                out.append([indent + prefix + str(key), val])
        return out
    return flatten(seq)


def wrap_rows(rows, max_rows=0):
    """
    Distribute the rows evenly across multiple columns until all are filled.
    """
    if not max_rows:
        return rows

    if len(rows) < max_rows:
        return rows

    for i in range(max_rows, len(rows)):
        rows[i % max_rows].extend(rows[i])

    return rows[:max_rows]


def format_str(text, length=None, pad=None):
    """
    Either pad or truncate a string to get it to the desired length
    """
    if not text or not length:
        return text

    if pad == True:
        pad = ' '

    if pad and len(text) < length:
        return text + pad * (length - len(text))

    if len(text) > length:
        return text[:length]

    return text


# This keeps instead of trims whitespace (for colored blocks)
tabulate.PRESERVE_WHITESPACE = True
