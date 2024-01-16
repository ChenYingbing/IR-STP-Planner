import torch

def test_torch_device():
  # Device configuration
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('set device=', device)

if __name__ == '__main__':
  test_torch_device()

  torch.zeros(1).cuda()
