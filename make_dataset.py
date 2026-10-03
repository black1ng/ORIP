from sklearn.datasets import load_iris

df = load_iris(as_frame=True).frame
df.to_csv("data/dataset.csv", index=False)
print(df.shape)