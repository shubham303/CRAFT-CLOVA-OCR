import statistics

# initializing list
test_list = [4, 5, 8, 9, 10]

# printing list
print("The original list : " + str(test_list))

# Standard deviation of list
# Using pstdev()
res = statistics.pstdev(test_list)
median = statistics.median(test_list)

degrees = [d for d in test_list if abs(d - median) <= res]

print(degrees)

# Printing result
print("Standard deviation of sample is : " + str(res))
print(mean)