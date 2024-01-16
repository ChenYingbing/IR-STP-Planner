from enum import Enum

class ColorPrinter:
  COLOR_MAP = {
    'red': '\x1b[0;30;41m',
    'green': '\x1b[0;30;42m',
    'yellow': '\x1b[0;30;43m',
    'blue': '\x1b[0;30;44m',
    'pink': '\x1b[0;30;45m',
    'white': '\x1b[0;30;47m',
  }

  @staticmethod
  def print(color_str: str, content: str, change_line=True):
    if change_line == True:
      print(ColorPrinter.COLOR_MAP[color_str] + content + '\x1b[0m')
    else:
      print(ColorPrinter.COLOR_MAP[color_str] + content + '\x1b[0m', end="")
