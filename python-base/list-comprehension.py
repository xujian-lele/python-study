list=[1,2,3,4]
print list
list1=[item*4 for item in list]
print list1
list2=[item*4 for item in list if item>2]
print list2

list3=[x*y for x in [1,2,3] for y in [1,2,3]]
print list3