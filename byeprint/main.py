from __future__ import annotations

import ast
from ast import AST, literal_eval
from collections import OrderedDict
from collections.abc import Generator, Iterable, Mapping
from contextlib import suppress
from dataclasses import is_dataclass
from enum import IntEnum
from inspect import ismethod
from io import StringIO
from pathlib import Path
from sys import _getframe
from textwrap import wrap
from typing import TYPE_CHECKING, Any
from unittest.mock import _Call

from executing import Source
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer

if TYPE_CHECKING:
  from collections.abc import Callable
  from types import FrameType

  from executing.executing import Executing

FORMATTER = Terminal256Formatter(style='monokai')
GENERATOR_TYPES = Generator, map, filter, zip, enumerate
INDENT = 2
LEXER = PythonLexer()
MISSING = object()
PARENTHESES_LOOKUP = [(list, '[', ']'), (set, '{', '}'), (frozenset, 'frozenset({', '})')]
PRETTY_KEY = '__prettier_formatted_value__'


def fmt(v: Any) -> Any:
  return {PRETTY_KEY: v}


def _style_as_int(v: Style | int) -> str:
  return str(v.value if isinstance(v, Style) else v)


def _as_ansi(s: str) -> str:
  return f'\033[{s}m'


def is_literal(s: Any) -> bool:
  try:
    literal_eval(s)
  except (TypeError, MemoryError, SyntaxError, ValueError):
    return False
  else:
    return True


class MetaLaxMapping(type):
  def __instancecheck__(cls, instance: Any) -> bool:
    return (
      hasattr(instance, '__getitem__')
      and hasattr(instance, 'items')
      and callable(instance.items)
      and type(instance) is not type
    )


class LaxMapping(metaclass=MetaLaxMapping):
  pass


class MetaDataClassType(type):
  def __instancecheck__(cls, instance: Any) -> bool:
    return is_dataclass(instance)


class DataClassType(metaclass=MetaDataClassType):
  pass


class SkipPrettyError(Exception):
  pass


class Formatter:
  def __init__(
    self,
    simple_cutoff: int = 10,
    width: int = 120,
  ) -> None:
    self._simple_cutoff = simple_cutoff
    self._width = width
    self._type_lookup: list[tuple[Any, Callable[[Any, str, int, int], None]]] = [
      (dict, self._dict),
      ((str, bytes), self._str_bytes),
      (tuple, self._tuples),
      ((list, set, frozenset), self._list_like),
      (bytearray, self._bytearray),
      (GENERATOR_TYPES, self._generator),
      (AST, self._ast_expression),
      (LaxMapping, self._dict),
      (DataClassType, self._dataclass),
    ]

  def __call__(self, value: Any, *, indent: int = 0, indent_first: bool = False) -> str:
    self._stream = StringIO()
    self._format(value, indent_current=indent, indent_first=indent_first)
    s = self._stream.getvalue()
    return highlight(s, lexer=LEXER, formatter=FORMATTER).rstrip('\n')

  def _format(self, value: Any, indent_current: int, *, indent_first: bool = False) -> None:
    if indent_first:
      self._stream.write(indent_current * ' ')
    pretty_func = getattr(value, '__pretty__', None)
    if ismethod(pretty_func) and not isinstance(value, _Call):
      try:
        gen = pretty_func(fmt=fmt, skip_exc=SkipPrettyError)
        self._render_pretty(gen, indent_current)
      except SkipPrettyError:
        pass
      else:
        return

    value_repr = repr(value)
    if len(value_repr) <= self._simple_cutoff and not isinstance(value, GENERATOR_TYPES):
      self._stream.write(value_repr)
    else:
      indent_new = indent_current + INDENT
      for t, func in self._type_lookup:
        if isinstance(value, t):
          func(value, value_repr, indent_current, indent_new)
          return

      self._raw(value, value_repr, indent_current, indent_new)

  def _render_pretty(self, gen: Iterable[Any], indent: int) -> None:
    prefix = False
    for v in gen:
      if isinstance(v, int) and v in {-1, 0, 1}:
        indent += v * INDENT
        prefix = True
      else:
        if prefix:
          self._stream.write('\n' + ' ' * indent)
          prefix = False

        pretty_value = v.get(PRETTY_KEY, MISSING) if (isinstance(v, dict) and len(v) == 1) else MISSING
        if pretty_value is not MISSING:
          self._format(pretty_value, indent)
        elif isinstance(v, str):
          self._stream.write(v)
        else:
          self._stream.write(repr(v))

  def _dict(self, value: Any, _: str, indent_current: int, indent_new: int) -> None:
    open_, before_, split_, after_, close_ = '{\n', indent_new * ' ', ': ', ',\n', '}'
    if isinstance(value, OrderedDict):
      open_, split_, after_, close_ = 'OrderedDict([\n', ', ', '),\n', '])'
      before_ += '('
    elif type(value) is not dict:
      open_, close_ = f'<{value.__class__.__name__}({{\n', '})>'

    self._stream.write(open_)
    for k, v in value.items():
      self._stream.write(before_)
      self._format(k, indent_new)
      self._stream.write(split_)
      self._format(v, indent_new)
      self._stream.write(after_)
    self._stream.write(indent_current * ' ' + close_)

  def _list_like(
    self,
    value: list[Any] | tuple[Any, ...] | set[Any],
    _: str,
    indent_current: int,
    indent_new: int,
  ) -> None:
    open_, close_ = '(', ')'
    for t, *oc in PARENTHESES_LOOKUP:
      if isinstance(value, t):
        open_, close_ = oc
        break

    self._stream.write(open_ + '\n')
    for v in value:
      self._format(v, indent_new, indent_first=True)
      self._stream.write(',\n')
    self._stream.write(indent_current * ' ' + close_)

  def _tuples(self, value: tuple[Any, ...], value_repr: str, indent_current: int, indent_new: int) -> None:
    if fields := getattr(value, '_fields', None):
      self._fields(value, zip(fields, value, strict=False), indent_current, indent_new)
    else:
      self._list_like(value, value_repr, indent_current, indent_new)

  def _str_bytes(
    self,
    value: str | bytes,
    value_repr: str,
    indent_current: int,
    indent_new: int,
  ) -> None:
    lines = list(self._wrap_lines(value, indent_new))
    if len(lines) > 1:
      self._str_lines(lines, indent_current, indent_new)
    else:
      self._stream.write(value_repr)

  def _str_lines(self, lines: Iterable[str | bytes], indent_current: int, indent_new: int) -> None:
    formatted = '\n'.join(indent_new * ' ' + repr(line) for line in lines)
    self._stream.write(f'(\n{formatted}\n{" " * indent_current})')

  def _wrap_lines(self, s: str | bytes, indent_new: int) -> Generator[str | bytes]:
    return (w for line in s.splitlines() for w in (wrap(str(line), self._width - indent_new - 3) or [str(line)]))

  def _generator(self, value: Generator[Any], _: str, indent_current: int, indent_new: int) -> None:
    name = value.__class__.__name__
    if name == 'generator':
      self._stream.write('(\n')
    else:
      self._stream.write(f'{name}(\n')
    for v in value:
      self._format(v, indent_new, indent_first=True)
      self._stream.write(',\n')
    self._stream.write(indent_current * ' ' + ')')

  def _bytearray(self, value: Any, _: str, indent_current: int, indent_new: int) -> None:
    self._stream.write('bytearray')
    lines = self._wrap_lines(bytes(value), indent_new)
    self._str_lines(lines, indent_current, indent_new)

  def _ast_expression(self, value: AST, _: str, indent_current: int, __: int) -> None:
    try:
      s = ast.dump(value, indent=INDENT)
    except TypeError:
      s = ast.dump(value)
    lines = s.splitlines(keepends=True)
    self._stream.write(lines[0])
    for line in lines[1:]:
      self._stream.write(indent_current * ' ' + line)

  def _dataclass(self, value: Any, _: str, indent_current: int, indent_new: int) -> None:
    try:
      field_items = value.__dict__.items()
    except AttributeError:
      field_items = ((f, getattr(value, f)) for f in value.__slots__)
    self._fields(value, field_items, indent_current, indent_new)

  def _raw(self, _: Any, value_repr: str, indent_current: int, indent_new: int) -> None:
    lines = value_repr.splitlines(keepends=True)
    if len(lines) > 1 or (len(value_repr) + indent_current) >= self._width:
      self._stream.write('(\n')
      wrap_at = self._width - indent_new
      prefix = indent_new * ' '

      for line in lines:
        sub_lines = wrap(line, wrap_at)
        for sline in sub_lines:
          self._stream.write(prefix + sline + '\n')
      self._stream.write(indent_current * ' ' + ')')
    else:
      self._stream.write(value_repr)

  def _fields(
    self,
    value: Any,
    fields: Iterable[tuple[str, Any]],
    indent_current: int,
    indent_new: int,
  ) -> None:
    self._stream.write(f'{value.__class__.__name__}(\n')
    for field, v in fields:
      self._stream.write(indent_new * ' ')
      if field:
        self._stream.write(f'{field}=')
      self._format(v, indent_new)
      self._stream.write(',\n')
    self._stream.write(indent_current * ' ' + ')')


class Style(IntEnum):
  reset = 0

  bold = 1
  not_bold = 22

  dim = 2
  not_dim = 22

  italic = 3
  not_italic = 23

  underline = 4
  not_underline = 24

  blink = 5
  not_blink = 25

  reverse = 7
  not_reverse = 27

  strike_through = 9
  not_strike_through = 29

  black = 30
  red = 31
  green = 32
  yellow = 33
  blue = 34
  magenta = 35
  cyan = 36
  white = 37

  bg_black = 40
  bg_red = 41
  bg_green = 42
  bg_yellow = 43
  bg_blue = 44
  bg_magenta = 45
  bg_cyan = 46
  bg_white = 47

  function = -1

  def __call__(self, inp: Any, *styles: Style | int | str, reset: bool = True) -> str:
    text = str(inp)
    codes = [
      _style_as_int(s.value if isinstance(s, Style) else s)
      if isinstance(s, Style | int)
      else _style_as_int(self.styles[s].value)
      for s in styles
    ]
    r = _as_ansi(';'.join(codes)) + text if codes else text
    if reset:
      r += _as_ansi(_style_as_int(self.reset))
    return r

  @property
  def styles(self) -> Mapping[str, Style]:
    return self.__class__.__members__

  def __str__(self) -> str:
    if self == self.function:
      return repr(self)
    return f'{self.__class__.__name__}.{self._name_}'


st = Style(-1)
fm = Formatter()


class ArgInfo:
  __slots__ = 'extra', 'name', 'value'

  def __init__(self, value: Any, *, name: str | None = None, **extra: Any) -> None:
    self.value = value
    self.name = name
    self.extra = []
    try:
      length = len(value)
    except TypeError:
      pass
    else:
      self.extra.append(('len', length))
    self.extra += [(k, v) for k, v in extra.items() if v is not None]


def join_args(args: list[ArgInfo]) -> str:
  lines = []
  for a in args:
    name_part = st(a.name, st.bold) if a.name and not is_literal(a.name) else ''
    type_part = st(a.value.__class__.__name__, st.blue, st.dim)
    value_part = fm(a.value)
    extras = ' '.join(f'{k}={v}' for k, v in a.extra)
    extras_part = f'{st(extras, st.dim)}' if extras else ''
    lines.append(f'{name_part}{st(":", st.dim)} {type_part} {st("=", st.red)} {value_part} {extras_part}\n')
  return '\n'.join(lines)


class ByePrint:
  arg_info = ArgInfo

  def __call__(
    self,
    *args: Any,
    file_: Any = None,
    flush_: bool = True,
    frame_depth_: int = 2,
    **kwargs: Any,
  ) -> Any:
    d_out = self._process(args, kwargs, frame_depth_)
    s = d_out
    print(s, file=file_, flush=flush_)
    if kwargs:
      return (*args, kwargs)
    if len(args) == 1:
      return args[0]
    return args

  def format(self, *args: Any, frame_depth_: int = 2, **kwargs: Any) -> str:
    return self._process(args, kwargs, frame_depth_)

  def _process(self, args: Any, kwargs: Any, frame_depth: int) -> str:
    try:
      call_frame: FrameType = _getframe(frame_depth)
    except ValueError:
      return join_args(list(self._args_inspection_failed(args, kwargs)))

    path = Path(call_frame.f_code.co_filename)
    if path.is_absolute():
      cwd = Path.cwd()
      with suppress(ValueError):
        path = path.relative_to(cwd)

    source = Source.for_frame(call_frame)
    ex = source.executing(call_frame)
    return join_args(
      list(self._process_args(ex, args, kwargs))
      if source.text and ex.node
      else list(self._args_inspection_failed(args, kwargs)),
    )

  def _args_inspection_failed(self, args: Any, kwargs: Any) -> Generator[ArgInfo]:
    yield from (self.arg_info(arg) for arg in args)
    yield from (self.arg_info(value, name=name) for name, value in kwargs.items())

  def _process_args(self, ex: Executing, args: Any, kwargs: Any) -> Generator[ArgInfo]:
    func_ast = ex.node
    atok = ex.source.asttokens()
    for arg, ast_arg in zip(args, func_ast.args, strict=False):
      yield self.arg_info(
        arg,
        name=(
          ast_arg.id if isinstance(ast_arg, ast.Name) else ' '.join(map(str.strip, atok.get_text(ast_arg).splitlines()))
        ),
      )
    kw_arg_names = {kw.arg: kw.value.id for kw in func_ast.keywords if isinstance(kw.value, ast.Name)}
    for name, value in kwargs.items():
      yield self.arg_info(value, name=name, variable=kw_arg_names.get(name))


p = ByePrint()
