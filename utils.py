import itertools


def build_kmers(sequence, ksize):
    r"""convert long seq to kmers"""
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers


def build_kmers_dict(k):
    r"""build kmer dict"""
    assert k <= 20
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    return {str(v): i for i, v in enumerate(kmers)}


struct_dict = {'.': 0, '(': 1, ')': 2}


def struct_encoding(struct):
    return [struct_dict[c] for c in struct]
