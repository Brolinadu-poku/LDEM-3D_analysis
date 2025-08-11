#Operator and Expressions: different types of operators in python

# Arithmetic Operators
print("\nArithmetic Operators:")
print("10+5 =", 10+5) # Addition
print("10-5=", 10-5)  # Subtraction
print("10*5=", 10*5) # Multiplication
print("10/5=", 10/5) # Division
print("10//5=", 10//5) # Floor division
print("10%3=", 10%3)  # Modulus
print("2**3=", 2**3) # power or exponential
''' In the above form, python outputs the operation and the results'''

#Assignment Operators
print("\nAssigment Operators:")
x = 5
print("x=", x)
x +=3 # Same as x = x + 3
print("x += 3 → ", x)
x *= 2 # same as x = x * 2
print("x *= 2 → ", x)

# Comparison of Operators
print("\nComparison of Operators:")
a = 10
b = 20
print ("a == b:", a == b) # This checks if the values of a and b are the same
print ("a != b:", a != b) # != is the "not equal" comparison operator. Thus, the operation means 'is a not equal to b?'. The prompt returns True if a and b are not the same, and False if they are the same. 
print("a > b:", a > b) # is a > b, if yes, returns true and if no, returns false. 
print("a < b:", a < b) # a < b, if yes ..............
print("a >= b:", a >= b) # asks if a is greater than or equal to b and .....
print("a <= b:", a <= b) # asks if a is less than or equal to .....

# Logical Operators
print("\nLogical Operators:") # returns the heading as Logical Operators
is_sunny = True # This a boolean that returns true if the weather is sunny. 
is_weekend = False #This is a boolean that returns false if it isn't weekend
print("is_sunny and is_weekend:", is_sunny and is_weekend) #returns false because is_weekend is false. for this operation to return true, both variables must be true. 
print("is_sunny or is_weekend:", is_sunny or is_weekend) # returns true if st least one is true. 
print("not is_sunny:", not is_sunny) # returns False because is_sunny is true. 

#Identity & Membership Operators
print("\nIdentity Operators:")
a = [1, 2, 3]
b = a 
c = [1, 2, 3]
print("a is b:", a is b) #returns true cos a=b, thus, they are the same objects
print("a is c:", a is c) #returns False  cos they are different objects
print("a == c", a==c) # returns True cos the contents are the same

# Membership Operators
print("\nMembership Operators:")
print("2 in [1, 2, 3]:", 2 in [1, 2, 3])
print("4 not in [1, 2, 3]:", 4 not in [1, 2, 3])
print(f"\nBrolin:") # a way of printing headers
