from src.model import Model

history = Model().train()
print(history.history.keys())
print("\n-----Successful-----\n")
