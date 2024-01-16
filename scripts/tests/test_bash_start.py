import argparse

if __name__ == '__main__':  
  parser = argparse.ArgumentParser()
  parser.add_argument("--tag", type=str, help="tag str to print", default="Default")
  args = parser.parse_args()

  for i in range(5):
    print("{}::debug_print({})".format(args.tag, i))
