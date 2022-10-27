from sklearn.preprocessing import StandardScaler
         
from sklearn import preprocessing, datasets

def main():
    X = [[0, 15],
        [1, -10]]
    # scale data according to computed scaling values
    output = StandardScaler().fit(X).transform(X)
    iris = datasets.load_iris()

    print(str(output))
    print(iris.DESCR)
    


if __name__ == "__main__":
    main()