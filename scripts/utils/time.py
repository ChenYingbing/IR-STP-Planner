from datetime import datetime

def get_date_str(format="%Y.%m.%d.%H_%M_%S"):
  return datetime.now().strftime(format)