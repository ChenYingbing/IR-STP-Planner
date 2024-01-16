#!/usr/bin/env python
from pickle import TRUE
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

FONT_STANDARD_SIZ = 8
TICK_FONT_LABEL_SIZ = 8
PLOT_LABEL_SIZ = 8

FONT_TYPE = 'Times New Roman'
FONT_STANDARD = {'family' : FONT_TYPE,
                  'weight' : 'normal', # bold
                  'size'   : FONT_STANDARD_SIZ,
                  }
TICK_LABEL_FONT = {'family' : FONT_TYPE,
                  'weight' : 'normal', # bold
                  'size'   : TICK_FONT_LABEL_SIZ,
                  }

PLOT_LABEL_FONT = {'family' : FONT_TYPE,
                  'weight' : 'normal', # bold
                  'size'   : PLOT_LABEL_SIZ,
                  }
TEXT_FONT = {'family' : FONT_TYPE,
             'weight' : 'normal', # bold
             'size'   : 8,
             }
# 210mm×297mm, 1 inch = 25.4mm, 0.707
# 8.226x11.69
def fig_reset(font_siz=None):
  # plt.rcParams.update({"font.family": FONT_TYPE})
  # plt.rcParams["font.family"] = FONT_TYPE
  plt.rc('text', usetex=True) # latex enabled
  plt.rc('font', family='serif')
  plt.rc('font', family='Times New Roman')

  if font_siz:
    plt.rcParams.update({"font.size": font_siz})
  else:
    plt.rcParams.update({"font.size": FONT_STANDARD_SIZ})
  plt.rcParams.update({"xtick.direction": 'in'})

def fig_set_text(locate_x, locate_y, text_str, 
                 ha="center", va="bottom", 
                 txt_font=TEXT_FONT, txt_color='k',
                 txt_rotation=0):
  plt.text(locate_x, locate_y, 
           text_str, ha=ha, va=va, color=txt_color,
           font=txt_font, rotation=txt_rotation)

def ax_set_text(ax, locate_x, locate_y, text_str, 
                ha="center", va="bottom", 
                txt_font=TEXT_FONT, txt_color='k',
                txt_rotation=0):
  ax.text(locate_x, locate_y, 
          text_str, ha=ha, va=va, color=txt_color,
          font=txt_font, rotation=txt_rotation)

def fig_set_text_box(locate_x, locate_y, text_str, 
                     ha="center", va="bottom", 
                     txt_font=TEXT_FONT, txt_color='k',
                     txt_rotation=0, face_color=(1., 1., 1.)):
  plt.text(locate_x, locate_y, 
           text_str, ha=ha, va=va, color=txt_color,
           font=txt_font, rotation=txt_rotation,
           bbox=dict(
             boxstyle="square",
             fc=face_color,
             ec="none"
           )   
  )

def fig_set_anotation(text, x0, y0, x1, y1, 
                      txt_x_offset = 0., 
                      txt_y_offset = 0.,
                      arrow='<->', color='k', 
                      set_txt_middle=True,
                      txt_rotation=0): 
  if set_txt_middle == False:
    plt.annotate(text, xy=(x0, y0), 
                 xytext=(x1, y1),
                 arrowprops=dict(arrowstyle=arrow, color=color))
  else:
    plt.annotate("", xy=(x0, y0), 
                 xytext=(x1, y1),
                 arrowprops=dict(arrowstyle=arrow, color=color))
    fig_set_text((x0+x1)*0.5+txt_x_offset, 
                 (y0+y1)*0.5+txt_y_offset, text, txt_color=color,
                 ha="center", va="center", txt_rotation=txt_rotation)

def subfig_reset():
  # fig = plt.figure()
  plt.xticks(fontproperties=FONT_TYPE, size=FONT_STANDARD_SIZ)
  plt.yticks(fontproperties=FONT_TYPE, size=FONT_STANDARD_SIZ)
  fig_reset()

def axis_set_title(ax_example, title_txt, loc_y=-0.15):
  ax_example.set_title(title_txt, y=loc_y, 
                       fontproperties=FONT_TYPE, fontsize=FONT_STANDARD_SIZ)

def axis_set_xticks(ax_example,
                    tick_values=None, 
                    tick_labels=None):
  if tick_values:
    ax_example.set_xticks(tick_values)
  if tick_labels:
    ax_example.set_xticklabels(tick_labels)

def axis_set_xlabel(ax_example, label: str, loc='right', 
                    labelpad=0, label_font=TICK_LABEL_FONT):
  if label:
    ax_example.set_xlabel(label, loc=loc, 
                          labelpad=labelpad, 
                          font=label_font)

def axis_set_yticks(ax_example,
                    tick_values=None,
                    tick_labels=None):
  if tick_values:
    ax_example.set_yticks(tick_values)
  if tick_labels:
    ax_example.set_yticklabels(tick_labels)

def axis_set_ylabel(ax_example,
                    label=None, loc='top', 
                    labelpad=0, label_font=TICK_LABEL_FONT):
  if label:
    ax_example.set_ylabel(label, loc=loc, 
                          labelpad=labelpad, 
                          font=label_font)

def axis_set_zlabel(ax_example,
                    label=None, labelpad=0, 
                    label_font=TICK_LABEL_FONT):
  if label:
    ax_example.set_zlabel(label, 
                          labelpad=labelpad, 
                          font=label_font)

def save_fig(file, vdpi=600):
  plt.savefig(file+'.pdf', bbox_inches='tight', 
              pad_inches=0.03, dpi=vdpi)

def save_eps(file, vdpi=600):
  plt.savefig(file+'.eps', bbox_inches='tight', dpi=vdpi)

def set_axis_equal():
  plt.gca().set_aspect('equal')
