import numpy as np
import matplotlib
from matplotlib import pyplot as plt 
import matplotlib.colors as mcolors

COLOR_RGB_MAP = {
  'black':      np.array([0, 0, 0]) / 255.,
  'white':      np.array([255, 255, 255]) / 255.,
  'grey':       np.array([50, 50, 50]) / 255.,
  'red':        np.array([255, 0, 0]) / 255.,
  'green':      np.array([0, 255, 0]) / 255.,
  'yellow':     np.array([255, 255, 0]) / 255.,
  'dark_green': np.array([0, 100, 0]) / 255.,
  'blue':       np.array([0, 0, 255]) / 255.,
  'light_blue': np.array([0, 255, 255]) / 255.,
  'orange':     np.array([255,165,0]) / 255.,
}

TWO_COLOR_SCHEME1 = [
  np.array([255., 59., 59.]) / 255., # red
  np.array([7., 7., 7.]) / 255., # black
]

TWO_COLOR_SCHEME2 = [
  np.array([254., 129., 125.]) / 255., # red
  np.array([129., 184., 223.]) / 255., # blue
]

THREE_COLOR_SCHEME1 = [
  np.array([210., 32., 39.]) / 255., # red
  np.array([56., 89., 137.]) / 255., # blue1
  np.array([127., 165., 183.]) / 255., # blue2
]

FOUR_COLOR_SCHEME1 = [
  np.array([43., 85., 125]) / 255., # Cblue
  np.array([69., 189., 155.]) / 255., # Cgreen
  np.array([240., 207., 110.]) / 255., # Cred: red
  np.array([253., 207., 110.]) / 255., # Cyel: yellow
]

FIVE_COLOR_SCHEME1 = [
  np.array([89., 89., 100.]) / 255., # purble black
  np.array([95., 198., 201.]) / 255., # light blue
  np.array([1., 86., 153.]) / 255., # blue
  np.array([250., 192., 15.]) / 255., # yellow
  np.array([243., 118., 74.]) / 255., # orange
]

FIVE_SEQ_COLOR_MAPS = [
  'Blues',
  'Greens',
  'Reds',
  'Greys',
  'Purples',
]

FIVE_MARKER_TYPES = [
  'o', '^', 's', 'X', ''
]

def color_from_index(index: int):
  '''
  Return color given index
  '''
  # BASE_COLORS, TABLEAU_COLORS, XKCD_COLORS
  color_dict = mcolors.TABLEAU_COLORS
  color_keys = list(color_dict.keys())
  color_num: int = len(color_dict)

  color_idx: int= (index+10) % color_num
  return color_dict[color_keys[color_idx]]

def plot_color_bar_example():
  fig, axs = plt.subplots(1, 2, subplot_kw=dict(projection="3d"))

  cmp = matplotlib.colors.ListedColormap([FIVE_COLOR_SCHEME1[0], FIVE_COLOR_SCHEME1[1],
                                          FIVE_COLOR_SCHEME1[2], FIVE_COLOR_SCHEME1[3],
                                          FIVE_COLOR_SCHEME1[4]])
  cmp_norm = matplotlib.colors.BoundaryNorm([0., 0.5, 1., 2., 4., 20.], cmp.N)
  fcb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=cmp_norm, cmap=cmp),
                     ax=axs, shrink=1.0, orientation='horizontal')
