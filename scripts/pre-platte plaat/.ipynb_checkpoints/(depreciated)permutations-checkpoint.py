import itertools

lst = [ 60, 70, 80, 90]

for subset in itertools.combinations(range(6, 10), 2):
    print( ' '.join(str(subset)))

permus = []

for indices in itertools.permutations(lst, 2):
    # print(indices)
    permus.append(indices)

print(permus)