import pandas as pd
import naive
import unrolled
import fast
import faster
from timeit import timeit


def main() -> None:
    df = pd.read_csv('data/sample.tsv', sep='\t').sample(frac=.5)
    #dot_naive = naive.sparsy(df.values).sum()
    #dot_unrolled = unrolled.sparsy(df.values).sum()
    #assert dot_naive == dot_unrolled, f'{dot_naive=} != {dot_unrolled=}'
    #print('naive: ', timeit(lambda: naive.sparsy(df.values), number=10))
    #print('unrolled: ', timeit(lambda: unrolled.sparsy(df.values), number=10))
    #fast.sparsy(df.values[:10, :])
    #faster.sparsy(df.values[:10, :])
    #print('fast: ', timeit(lambda: fast.sparsy(df.values), number=1))
    #print('faster: ', timeit(lambda: faster.sparsy(df.values), number=1))

    

if __name__ == '__main__':
    main()
