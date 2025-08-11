#String
name = "Brolin"
language = 'python'
print(name, language)

#integer and float
age = 26
height = 5.6
print(age, height)

#Boolean
learning = True
is_joking = False
print (learning, is_joking)

# NoneType
middle_name = None
print(middle_name)

# Multiline String - Allows you to write a string that spans multiple lines
message = '''Hello, 
This is a multi-line sting.'''
print (message)

#Docstrings: inside functions, classes, or modules,triple quotes are used to describe what they do:
def greet(name):
    '''This function greets the user by name'''
    print("Hello", name)

'''
type() function helps check the data type of any variable.
'''

print(type(name))  #<class 'str'>
print(type(age))   #<class 'int'>
print(type(height)) #<class 'float>
print(type(learning)) #<class 'bool'>
print(type(middle_name)) #<class 'NoneType'>

#int to float: changing an integer to a float
x = 5
y = float(x)
print(y, type(y))

#float to integer(i.e. decimal part is dropped)
a = 9.7
b = int(a)
print(b, type(b))

#number  to string
num = 100
text = str(num)
print(text, type(text))

#string to int (if string contains a number)
s ="50"
n =int(s)
print(n, type(n))

'''
Note that, we can't convert a non-numeric string to an integer. 
'''