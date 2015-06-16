list1=[1,2,3,4]
set1=set(list1)
print set1
print 'loop set'
for item in set1:
    print item
print 'loop list'
for item in list1:
    print item 
    
print 'loop dictionary'
d1={}
d1[1]=1
d1[2]=2
d1[3]=3
d1[4]=4

for item in d1:
    print d1[item]