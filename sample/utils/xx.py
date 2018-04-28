a = [3,4,1,7,2]

b=sorted(enumerate(a), key=lambda x:x[1],reverse=True)
c1 = [x[1] for x in b]
c2 = [x[0] for x in b]

print(c1)
print(c2)