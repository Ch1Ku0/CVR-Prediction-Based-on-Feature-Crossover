import model
import torch


def test_model():

  feature_vocab = {"A": 10, "B": 12, "C": 20, "D": 15, "E": 5}
  # feature_vocab = {"A": 10, "B": 12, "C": 20}
  embedding_size = 128
  m = model.PLE(feature_vocab, embedding_size)
  inputs = {
      "A": torch.tensor([[1], [2]]),
      "B": torch.tensor([[2], [3]]),
      "C": torch.tensor([[10], [11]]),
      "D": torch.tensor([[11], [13]]),
      "E": torch.tensor([[2], [3]])
  }
  click, conversion = m(inputs)
  print("click_pred:", click,click.shape)
  print("covnersion_pred:", conversion, conversion.shape)

  click_label = torch.tensor([1.0, 1.0])
  conversion_label = torch.tensor([1.0, 0.0])

  loss = m.loss(click_label, click, conversion_label, conversion)
  print("loss: ", loss)


if __name__ == "__main__":
  test_model()
